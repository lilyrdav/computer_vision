from fastai.vision.all import *

from kornia.filters.blur_pool import MaxBlurPool2D
from torch import nn

from fastai.callback.progress import ProgressCallback
from fastai.callback.schedule import lr_find, fit_flat_cos, fit_one_cycle

from fastai.data.core import Datasets
from fastai.data.external import untar_data, URLs
from fastai.data.transforms import Categorize, GrandparentSplitter, IntToFloatTensor, Normalize, ToTensor, parent_label

from fastai.layers import Mish
from fastai.learner import Learner

from fastai.metrics import LabelSmoothingCrossEntropy, top_k_accuracy, accuracy

from fastai.optimizer import ranger, Lookahead, RAdam

from fastai.vision.augment import FlipItem, RandomResizedCrop, Resize
from fastai.vision.core import PILImage, imagenet_stats, get_image_files
from fastai.vision.models.xresnet import xresnet50

# https://discuss.pytorch.org/t/how-can-i-replace-an-intermediate-layer-in-a-pre-trained-network/3586/7
import kornia
def convert_MP_to_blurMP(model, layer_type_old):
    conversion_count = 0
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = convert_MP_to_blurMP(module, layer_type_old)

        if type(module) == layer_type_old:
            layer_old = module
            layer_new = MaxBlurPool2D(3, True)
            model._modules[name] = layer_new

    return model

def main():

    path = untar_data(URLs.IMAGEWOOF_320)

    lbl_dict = dict(
    n02086240= 'Shih-Tzu',
    n02087394= 'Rhodesian ridgeback',
    n02088364= 'Beagle',
    n02089973= 'English foxhound',
    n02093754= 'Australian terrier',
    n02096294= 'Border terrier',
    n02099601= 'Golden retriever',
    n02105641= 'Old English sheepdog',
    n02111889= 'Samoyed',
    n02115641= 'Dingo'
    )
    tfms = [[PILImage.create], [parent_label, lbl_dict.__getitem__, Categorize()]]
    item_tfms = [ToTensor(), Resize(128)]
    batch_tfms = [FlipItem(), RandomResizedCrop(128, min_scale=0.35),
                IntToFloatTensor(), Normalize.from_stats(*imagenet_stats)]


    items = get_image_files(path)
    split_idx = GrandparentSplitter(valid_name='val')(items)

    dsets = Datasets(items, tfms, splits=split_idx)
    dls = dsets.dataloaders(after_item=item_tfms, after_batch=batch_tfms, bs=64)

    # arch = xresnet50(pretrained=False, act_cls=Mish, sa=True)

    # arch[0]

    # net = xresnet34(pretrained=True, n_out=10)
    # net = convert_MP_to_blurMP(net, nn.MaxPool2d)

    learn = Learner(dls, xresnet34, metrics=[top_k_accuracy, accuracy])
    # learn.fine_tune(1)

    learn.show_results()

if __name__=="__main__": 
    main() 