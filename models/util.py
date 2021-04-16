from fastai.vision.all import *
import pandas as pd


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
            ReduceLROnPlateau() # If the error rate plateaus then reduce it by a factor of 10
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


def chexpert_data_loader(reparse=False, img_size=224, bs=32):
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
        cat_df = pd.concat([train_df, valid_df]).fillna(0).replace(-1, 0)

        labels = list(cat_df.iloc[:,6:].columns.values)

        # we resize so that the larger dimension is match and crop 
        # (randomly on the training set, center crop for the validation set)
        # Using batchsize of 128 for trainng and validation
        dls = ImageDataLoaders.from_df(
        df=cat_df, path=chexpert, folder='/storage/archive/',
        label_col=labels, y_block=MultiCategoryBlock(encoded=True, vocab=labels),
        item_tfms=Resize(img_size), bs=bs, val_bs=bs)

        torch.save(dls, saved_pkl)
    
    return dls, labels
