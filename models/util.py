from fastai.vision.all import *
import pandas as pd


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


    def learn_model(self, use_saved=False, train_saved=False,
                    # other args for Learner.fine_tune
                    **kwargs):
        """ Load a previously saved model or train a new model """

        saved_model_name = f'{self.learn.arch.__name__}-chexpert'

        if use_saved:
            try:
                self.learn.load(saved_model_name)
                if not train_saved: return
                else: self.learn.loss_func = self.loss_func
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
            SaveModelCallback(fname=saved_model_name, with_opt=True, monitor='accuracy_multi'), # Save the model if it improves
            ReduceLROnPlateau(monitor='accuracy_multi'), # If the error rate plateaus then reduce it by a factor of 10
            CSVLogger(f'{saved_model_name}.csv'), # CSV file for training results
            EarlyStoppingCallback(monitor='accuracy_multi', patience=5),
        ]
        
        self.learn.fine_tune(cbs=callbacks, base_lr=self.base_lr, **kwargs)


def chexpert_data_loader(reparse=True, bs=32):
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
    
    if saved_pkl.exists() and not reparse:
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

        # train_df = train_df.fillna(-1).applymap(lambda l: l if l != -1 else random.uniform(0.8, 1))
        # valid_df = train_df.fillna(-1).replace(-1, 1)
        
        cat_df = (pd.concat([train_df, valid_df]).fillna(-1)
                  .applymap(lambda l: l if l != -1 else random.uniform(0.8, 1)))

        labels = list(cat_df.iloc[:,6:].columns.values)
        
        # Random vertical flipping (L-R), Resize to 256x256, Crop and Resize to 224x224
        # The random aspects of the transforms only apply to training ds
        item_tfms = Resize(256)
        batch_tfms = [
            Flip(),
            RandomResizedCrop(224),
        ]

        # Data is auto normalized to ImageNet ds
        dls = ImageDataLoaders.from_df(
        df=cat_df, path=chexpert, folder='/storage/archive/',
        label_col=labels, y_block=MultiCategoryBlock(encoded=True, vocab=labels),
        item_tfms=item_tfms, batch_tfms=batch_tfms, bs=bs, val_bs=bs)

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

hierarchy_map = ([2,4,5,6,7,8], [1,3,3,3,6,3])

@delegates()
class BCEFlatHLCP(BaseLoss):
    "Uses Hierarchical Label Conditional Probability with BCELossFlat"
    @use_kwargs_dict(keep=True, weight=None, reduction='mean', pos_weight=None)
    def __init__(self, *args, axis=-1, floatify=True, hierarchy_map=None, **kwargs):
        super().__init__(nn.BCELoss, *args, axis=axis, floatify=floatify, is_2d=False, **kwargs)
        self.orig_indices = hierarchy_map[0]
        self.mask_indices = hierarchy_map[1]

    def __call__(self, inp, targ, **kwargs):
        "Here we apply hierarchy to the inputs and targets"
        modified_inp = inp.detach().sigmoid()
        modified_targ = torch.round(targ)
        
        modified_inp[:, self.orig_indices] = torch.mul(modified_inp[:, self.orig_indices], modified_targ[:, self.mask_indices])
        modified_targ[:, self.orig_indices]= torch.mul(modified_targ[:, self.orig_indices], modified_targ[:, self.mask_indices])
        
        modified_inp.requires_grad = True
        
        return super().__call__(modified_inp, modified_targ, **kwargs)
