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
        self.learn = cnn_learner(dls, arch, **kwargs)
    
    def learn_model(self, use_saved=False,                
                    # other args for Learner.fine_tune
                    **kwargs):
        """ Load a previously saved model or train a new model """

        saved_model_name = f'{self.learn.arch.__name__}-chexpert'

        if use_saved:
            try:
                self.learn.load(saved_model_name)
                return
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
            SaveModelCallback(fname=saved_model_name), # Save the model if it improves
            ReduceLROnPlateau(), # If the error rate plateaus then reduce it by a factor of 10
            CSVLogger(f'{saved_model_name}.csv') # CSV file for training results
        ]
        
        self.learn.fine_tune(cbs=callbacks, **kwargs)


def get_predictions(preds, vocab, thresh=0.15):
    """Returns predictions >= thresh"""
    pred_labels = []
    pred_tuples = []
    prob_preds = torch.sigmoid(preds[0])
    mask = prob_preds >= thresh
    significant_labels = vocab[mask]
    significant_preds = prob_preds[mask]
    
    for i in range(len(significant_labels)):
        pred_tuples.append((significant_labels[i], significant_preds[i].item()))
    
    pred_tuples.sort(key = lambda x: x[1])
    
    for tup in reversed(pred_tuples):
        pred_labels.append(f'{tup[0]}: {tup[1]:.3f}')
    
    return pred_labels


def chexpert_data_loader(reparse=False, img_size=256, bs=64):
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

        # we resize so that the larger dimension is match and crop 
        # (randomly on the training set, center crop for the validation set)
        dls = ImageDataLoaders.from_df(
        df=cat_df, path=chexpert, folder='/storage/archive/',
        label_col=labels, y_block=MultiCategoryBlock(encoded=True, vocab=labels),
        item_tfms=Resize(img_size), bs=bs, val_bs=bs)

        torch.save(dls, saved_pkl)
    
    return dls, labels
