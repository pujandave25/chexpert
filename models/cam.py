'''
Multi-class Multi-label Grad-CAM algorithm.

Inspired from the multi-class single-label algorithm presented in the following paper:

    Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D. and Batra, D., 2017.
    Grad-cam: Visual explanations from deep networks via gradient-based localization.
    In Proceedings of the IEEE international conference on computer vision (pp. 618-626).
'''

from fastai.vision.all import *
import pandas as pd
import util


def _get_last_layer_activations(model, loss_fn, x, classes):
    with hook_output(model[0]) as hook_acts:
        with hook_output(model[0], grad=True) as hook_grads:
            preds = model(x)
            loss_fn(preds[0], classes).backward()
    return hook_acts, hook_grads, preds


def _show_activations_heatmap(model, learn, idx, ax1, ax2):
    image, classes = learn.dls.valid_ds[idx]
    
    # Transform the image as per the dataloader
    # split_idx=1 for validation and test set (center crop)
    image = learn.dls.after_item[0](image, split_idx=1)
    true_labels = learn.dls.vocab[classes == 1]
    x, = first(learn.dls.test_dl([image]))
    classes = classes.cuda()
    x = x.cuda()

    hook_acts, hook_grads, preds = _get_last_layer_activations(model, learn.loss_func, x, classes)
    acts = hook_acts.stored[0].cpu()
    grads = hook_grads.stored[0][0].cpu()
    grads_chan = grads.mean(1).mean(1)
    mult = F.relu((acts * grads_chan[...,None,None]).mean(0))
        
    pred_labels = util.get_predictions(preds, learn.dls.vocab, thresh=0.15)
    
    image.show(ax1, extent=(0,56,56,0))
    image.show(ax2, extent=(0,56,56,0))
    
    ax2.imshow(mult, alpha=0.6, extent=(0,56,56,0),
              interpolation='bilinear', cmap='magma')
    
    title = f'True: {true_labels}\n    Pred: {pred_labels}\n'
    
    return title


def plot_cam(learn):
    model = learn.model.eval()
    model = model.cuda()
    val_size = len(learn.dls.valid_ds)

    fig, ax = plt.subplots(3, 3*2)
    fig.set_size_inches(12,6)
    
    title = ''

    for i in range(3):
        for j in range(0, 6, 2):
            idx = random.randint(0, val_size)
            count = 3*i + int(j/2)
            image_id = chr(ord('A') + count)
            sub_title = _show_activations_heatmap(model, learn, idx, ax[i, j], ax[i, j+1])
            ax[i, j].set_title(f'{image_id}')
            ax[i, j+1].set_title(f'{image_id}-CAM')
            title += f'{image_id}. {sub_title}\n'
    
    fig.text(0.15, -0.65, title)

    plt.show()
