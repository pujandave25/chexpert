from fastai.vision.all import *
import pandas as pd
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
from sklearn.metrics import roc_auc_score


def _accum_values(self, preds, targs,learn=None):
    "Store targs and preds"
    to_d = learn.to_detach if learn is not None else to_detach
    preds,targs = to_d(preds),to_d(targs)
    if self.flatten: preds,targs = flatten_check(preds,targs)
    targs = torch.round(targs)
    self.preds.append(preds)
    self.targs.append(targs)
    

AccumMetric.accum_values = _accum_values

        
class ChexpertLearner(Learner):
    """ Learner wrapper specifically
        created for CheXpert
    """


    def __init__(self, dls, arch, **kwargs):
        # For guide on all the input parameters,
        # check doc for cnn_learner
        self.path = Path('../saves/')
        self.learn = cnn_learner(dls, arch, path=self.path, **kwargs)
        self.loss_func = kwargs.get('loss_func')
        self.base_lr = 0.002


    def find_lr(self):
        # Quick way to find the optimal LRs
        # take the max of the lr_min (min loss/10)
        # and lr_steep (steepest loss/lr curve) as
        # fine_tune will use a cycle rangining from
        # base_lr/100 to base_lr

        # Refer: https://iconof.com/1cycle-learning-rate-policy/
        # Citation:
        #     Smith LN. Cyclical learning rates for training neural networks.
        #     In 2017 IEEE winter conference on applications of computer vision
        #     (WACV) 2017 Mar 24 (pp. 464-472). IEEE.
        
        torch.cuda.empty_cache()

        lr_min, lr_steep = self.learn.lr_find()
        self.base_lr = max(lr_min, lr_steep)
        print(f'lr_min/10: {lr_min}, lr_steep: {lr_steep}, base_lr: {self.base_lr}')


    def learn_model(self, use_saved=False, train_saved=False, old_learner=None,
                    # other args for Learner.fine_tune
                    **kwargs):
        """ Load a previously saved model or train a new model """

        saved_model_name = f'{self.learn.arch.__name__}-chexpert'

        if use_saved:
            try:
                if not train_saved:
                    self.learn.load(saved_model_name)
                    return
                if old_learner:
                    old_learner.load(saved_model_name)
                    self.learn.model[0].load_state_dict(old_learner.model[0].state_dict())
                    self.learn.loss_func = self.loss_func
                    
            except FileNotFoundError as e:
                print(f'Could not find saved model {saved_model_name}.')
        
        torch.cuda.empty_cache()

        # `fine_tune` first freezes the body and then updates only head weights for freeze_epochs
        # then it unfreezes the body and updates all weights for epochs.
        # Internally it uses `fit_one_cycle` for cyclical learning rates:
        # Refer: https://iconof.com/1cycle-learning-rate-policy/
        # Citation:
        #     Smith LN. Cyclical learning rates for training neural networks.
        #     In 2017 IEEE winter conference on applications of computer vision
        #     (WACV) 2017 Mar 24 (pp. 464-472). IEEE.
        
        # Using callbacks for a few things:
        callbacks = [
            ShowGraphCallback(), # Show the graph
            SaveModelCallback(fname=saved_model_name, with_opt=True), # Save the model if it improves
            ReduceLROnPlateau(patience=2), # If the validation loss stops improving then reduce it by a factor of 10
            CSVLogger(f'{saved_model_name}.csv'), # CSV file for training results
            EarlyStoppingCallback(patience=5),
        ]
        
        self.learn.fine_tune(cbs=callbacks, base_lr=self.base_lr, **kwargs)


def chexpert_data_loader(reparse=True, bs=32, use_hierarchy=False):
    """ Load the CheXpert dataset.
        Try loading from the saved chexpert-small.pkl
        if it exists and reparse is not requested.
        If the data is reparsed then save it to chexpert-small.pkl
        
        Following params only used if reparsing:
            
            img_size: Resize the images to size x size
            bs: batch size
        
        NOTE: This assumes that the raw dataset is located at
                /storage/archive/CheXpert-v1.0-small or as a
                zip file /storage/archive/CheXpert-v1.0-small.zip        
    """
    
    chexpert = Path('/storage/archive/CheXpert-v1.0-small')
    saved_pkl = Path('../dataset/chexpert-small.pkl')
    hc_saved_pkl = Path('../dataset/chexpert-small-hc.pkl')
    
    if ((not use_hierarchy and saved_pkl.exists()) or (use_hierarchy and hc_saved_pkl.exists())) and not reparse:
        if use_hierarchy:
            dls = torch.load(hc_saved_pkl)
        else:
            dls = torch.load(saved_pkl)
        labels = list(dls.items.iloc[:,6:].columns.values)
    else:
        if not chexpert.exists():
            file_extract(chexpert.parent/(chexpert.name +'.zip'))
            
        train_df = pd.read_csv(chexpert/'train.csv')
        valid_df = pd.read_csv(chexpert/'valid.csv')
        train_df.insert(loc=1, column='Valid', value=False)
        valid_df.insert(loc=1, column='Valid', value=True)
        
        # We replace all unavailable or -1 labels with a number closer to 1
        # this is the Label Smoothing Regularization (LSR) approach.
        # LSR is only applied to training set, for validation set we use 1.

        # Only apply label smoothing when working with full dataset
        cat_df = (pd.concat([train_df, valid_df]).fillna(-1)
                  .applymap(lambda l: l if l != -1 else (
                      0 if use_hierarchy else random.uniform(0, 0.001))))
        
        labels = list(cat_df.iloc[:, 6:].columns.values)
        
        if use_hierarchy:
            # As the model is to be focused on Atelectasis, Cardiomegaly,
            # Consolidation, Edema, and Pleural Effusion, for hierarchy,
            # let us just focus on a single hierarchy involving
            # 'Lung Lesion',
            # 'Edema',
            # 'Pneumonia',
            # 'Atelectasis',
            # With 'Lung Opacity' and 'Consolidation' as parents.
            # We only pick samples with the parents as positive.
            # This approximates the conditional probability behavior
            
            col_indices_to_keep = list(range(6)) + list(range(10,15))
            col_indices_to_keep.remove(12) # Remove 'Consolidation'
            cat_df = (cat_df[(cat_df['Lung Opacity'] > 0) & (cat_df['Consolidation'] > 0)]
                      .iloc[:, col_indices_to_keep])

            # Remove lines where no class is True to avoid ROC metric exception
            cat_df = cat_df[cat_df.iloc[:, 6:].sum(1) > 0]
            labels = list(cat_df.iloc[:, 6:].columns.values)
            

        # Random vertical flipping (L-R), Resize to 256x256, Crop and Resize to 224x224
        # The random aspects of the transforms only apply to training ds
        item_tfms = Resize(256)
        batch_tfms = [
            Flip(),
            # During test runs, it was noticed that random cropping is not helping much
            # RandomResizedCrop(224),
        ]

        # Data is auto normalized to ImageNet ds
        dls = ImageDataLoaders.from_df(
        df=cat_df, path=chexpert, folder='/storage/archive/',
        label_col=labels, y_block=MultiCategoryBlock(encoded=True, vocab=labels),
        item_tfms=item_tfms, batch_tfms=batch_tfms, bs=bs, val_bs=bs)

        if use_hierarchy:
            torch.save(dls, hc_saved_pkl)
        else:
            torch.save(dls, saved_pkl)
    
    return dls, labels


# Loss function with Hierarchical Label Conditional Probability
#
# Source1: http://proceedings.mlr.press/v102/chen19a/chen19a.pdf
#
# Source2:
#  J. Irvin, P. Rajpurkar, M. Ko, Y. Yu, S. Ciurea-Ilcus, C. Chute,
#  H. Marklund, B. Haghgoo, R. L. Ball, K. Shpanskaya, J. Seekins,
#  D. A. Mong, S. S. Halabi, J. K. Sandberg, R. Jones, D. B. Larson,
#  C. P. Langlotz, B. N. Patel, M. P. Lungren, A. Y. Ng,
#  CheXpert: A large chest radiograph dataset with uncertainty
#  labels and expert comparison, in: AAAI, 2019.
#
# Source3: https://arxiv.org/pdf/1911.06475.pdf

# The below single ancestor hierarchy works as the labels are listed in order
# of their level, so we would have already checked the ancestors of ancestor.
# hierarchy_map = [-1, # 'No Finding',
#                  -1, # 'Enlarged Cardiomediastinum',
#                  1,  # 'Cardiomegaly',
#                  -1, # 'Lung Opacity',
#                  3,  # 'Lung Lesion',
#                  3,  # 'Edema',
#                  3,  # 'Consolidation',
#                  6,  # 'Pneumonia',
#                  3,  # 'Atelectasis',
#                  -1, # 'Pneumothorax',
#                  -1, # 'Pleural Effusion',
#                  -1, # 'Pleural Other',
#                  -1, # 'Fracture',
#                  -1] # 'Support Devices'

# Class index and ancestor index
HIERARCHY_MAP = ([2,4,5,6,7,8], [1,3,3,3,6,3])

@delegates()
class BCEFlatHLCP(BaseLoss):
    "Uses Hierarchical Label Conditional Probability with BCELossFlat"
    @use_kwargs_dict(keep=True, weight=None, reduction='mean', pos_weight=None)
    def __init__(self, *args, axis=-1, floatify=True, **kwargs):
        super().__init__(nn.BCELoss, *args, axis=axis, floatify=floatify, is_2d=False, **kwargs)
        self.orig_indices = HIERARCHY_MAP[0]
        self.mask_indices = HIERARCHY_MAP[1]

    def __call__(self, inp, targ, **kwargs):
        "Here we apply hierarchy to the inputs and targets"
        modified_inp = inp.detach().sigmoid()
        modified_targ = torch.round(targ)
        
        modified_inp[:, self.orig_indices] = torch.mul(modified_inp[:, self.orig_indices], modified_targ[:, self.mask_indices])
        modified_targ[:, self.orig_indices]= torch.mul(modified_targ[:, self.orig_indices], modified_targ[:, self.mask_indices])
        
        modified_inp.requires_grad = True
        
        return super().__call__(modified_inp, modified_targ, **kwargs)

class DAM:
    def __init__(self, model_dam, dls, folder, lr=0.1, gamma=500, weight_decay=0, margin=1.0):
        self.model = model_dam.cuda()
        self.dls = dls
        self.folder = folder
        
        self.loss_func = AUCMLoss()
        self.opt_func = PESG(
            self.model,
            a=self.loss_func.a, 
            b=self.loss_func.b, 
            alpha=self.loss_func.alpha, 
            lr=lr, 
            gamma=gamma, 
            margin=margin
        )
    
    def eval(self):
        self.model.eval()
        test_pred = []
        test_true = [] 
        for j, (test_data, test_targets) in enumerate(self.dls.valid):
            y_pred = self.model(test_data)
            test_pred.append(y_pred.cpu().detach().numpy())
            test_true.append(test_targets.cpu().detach().numpy())
        test_true = np.concatenate(test_true)
        test_true[test_true < 0.5] = 0
        test_true[test_true >= 0.5] = 1
        test_pred = np.concatenate(test_pred)
        return roc_auc_score(test_true, test_pred, average='weighted')
    
    def train(self, max_epoch=3, lr_div=2, checkpoint=None):
        # Load checkpoint if available
        if checkpoint != None:
            model_checkpoint = torch.load(self.folder/checkpoint)
            self.model.load_state_dict(model_checkpoint['state_dict'])
            self.opt_func.load_state_dict(model_checkpoint['optimizer'])

        # Train model
        max_auc = 0

        for epoch in range(max_epoch):

            train_pred = []
            train_true = []
            self.model.train()    
            for i, (data, targets) in enumerate(self.dls.train):
                self.opt_func.zero_grad()
                y_pred = self.model(data)
                loss = self.loss_func(y_pred, targets)
                loss.backward(retain_graph=True)
                self.opt_func.step()

                train_pred.append(y_pred.cpu().detach().numpy())
                train_true.append(targets.cpu().detach().numpy())

            self.opt_func.lr = self.opt_func.lr/lr_div
            self.opt_func.update_regularizer()

            train_true = np.concatenate(train_true)
            train_true[train_true < 0.5] = 0
            train_true[train_true >= 0.5] = 1
            train_pred = np.concatenate(train_pred)
            
            train_auc = roc_auc_score(train_true, train_pred, average='weighted') 

            # Eval model
            val_auc =  self.eval()

            # print results
            print("epoch: {}, train_loss: {:4f}, train_auc:{:4f}, test_auc:{:4f}, lr:{:4f}".format(epoch, loss.item(), train_auc, val_auc, self.opt_func.lr ))

            # Save checkpoint
            if val_auc > max_auc:
                max_auc = val_auc
                torch.save({
                    'epoch': epoch+1,
                    'best_auc': val_auc,
                    'state_dict': self.model.state_dict()
                }, self.folder/f"m-epoch {epoch+1}-{time.strftime('%Y_%h_%d-%H_%M_%S')}.pth.tar")
