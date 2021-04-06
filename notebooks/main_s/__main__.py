#%%

import fastbook
fastbook.setup_book()

#%%

from fastbook import *
from fastai.vision.widgets import *
from fastai.callback.tensorboard import *

IN_NOTEBOOK = False

#%%

import torch
torch.cuda.empty_cache()

#%%

import datetime
import pandas as pd
import numpy as np

#%%

path = Path('../data/train')

#%%

path.ls()

#%%

def sort_image_by_dimensions(path, size):
    w, h = Image.open(path).size
    return w >= size and h >= size

#%%

image_size = 460

#%%

def correct_sized_images(path):
    return [create_image(img) for img in get_image_files(path) if sort_image_by_dimensions(img, image_size)]

def create_image(image_path):
    res = Resize(image_size)
    res.pcts = (0.5,0.5)
    image = res.encodes(PILImage.create(image_path))
    img = PILImage.create("")
    img
    

#%%

def get_y(r): return parent_label(r).split(" ")

#%%

kwargs = {'num_workers': 1, 'pin_memory': True}

def get_dls(bs, size) -> DataLoaders:
    dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                        splitter=RandomSplitter(seed=42),
                        get_items=correct_sized_images,
                        get_y=get_y,
                        batch_tfms=aug_transforms(size=size, min_scale=0.75))

    return dblock.dataloaders(path, bs=bs, num_workers=0).cuda()

#%%

batch_size = 64
dls = get_dls(batch_size, 64)

#%%

# dls.show_batch(max_n=4, nrows=1)

#%%

cuda0 = torch.device('cuda:0')
cpu = torch.device('cpu')

#%%

from fastai.losses import *
from torch.nn.modules import loss

class smooth_binary_cross_entropy(loss._Loss):

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean',
             pos_weight: Optional[Tensor] = None) -> None:
        super(smooth_binary_cross_entropy, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        c = target.shape[1]
        eps = 0.1
        smoothed_target = torch.where(target==1, 1-(eps+(eps/c)), eps/c)
        return F.binary_cross_entropy_with_logits(input,
                                                  smoothed_target,
                                                  pos_weight=self.pos_weight,
                                                  reduction=self.reduction)

@delegates()
class smooth_loss_v2(BaseLoss):
    "Same as `nn.BCEWithLogitsLoss`, but flattens input and target."
    @use_kwargs_dict(keep=True, weight=None, reduction='mean', pos_weight=None)
    def __init__(self, *args, axis=-1, floatify=True, thresh=0.5, **kwargs):
        if kwargs.get('pos_weight', None) is not None and kwargs.get('flatten', None) is True:
            raise ValueError("`flatten` must be False when using `pos_weight` to avoid a RuntimeError due to shape mismatch")
        if kwargs.get('pos_weight', None) is not None: kwargs['flatten'] = False
        super().__init__(smooth_binary_cross_entropy, *args, axis=axis, floatify=floatify, is_2d=False, **kwargs)
        self.thresh = thresh

    def decodes(self, x):    return x>self.thresh
    def activation(self, x): return torch.sigmoid(x)

#%%

pos_weight = torch.ones(3, device=cuda0)

#%%



#%%

#learn.lr_find()

#%%

class FineTuner():

    def __init__(self, learner: Learner):
        self.learner = learner

    def fit_with_presizing(self,
                           presizing_cycles: int,
                           max_image_size: int,
                           image_sizes: Optional[list] = None,
                           epochs_per_cycle: Optional[list] = None,
                           batch_size: int = 64):

        if isinstance(image_sizes, type(None)):
            image_sizes = [int(max_image_size/(presizing_cycles - i)) for i in range(presizing_cycles)]

        if isinstance(epochs_per_cycle, type(None)):
            epochs_per_cycle = [5]

        for i in range(presizing_cycles):
            image_size_cycle = self.get_first_index_else_last(image_sizes, i, presizing_cycles)
            epochs = self.get_first_index_else_last(epochs_per_cycle, i, presizing_cycles)

            print(f"Fine tuning cycle '{i + 1}' for {epochs} epochs with parameters\n"
                  f"\tbatch_size: {batch_size},\n"
                  f"\timage_size: {image_size_cycle}:")

            self.learner.dls = get_dls(batch_size, image_size_cycle)
            self.fine_tune_with_epoch(epochs)
            print("")

    def fine_tune_with_epoch(self, epochs: int):
            self.learner.fit_one_cycle(epochs)

    def get_first_index_else_last(self, list_obj: list, idx: int, total_iters: int):
        item = list_obj[0]

        if idx == total_iters - 1:
            item = list_obj[-1]

        return list_obj[idx] if idx < len(list_obj) - 1 else item

#%%

learn = cnn_learner(dls,
                        resnet50,
                        metrics=partial(accuracy_multi, thresh=0.2))

epochs, lr = 1, 1e-2
if __name__ == '__main__':
    # learn.fit_one_cycle(epochs, lr)
    
    learn.fit_one_cycle(epochs, lr)
    
    # tuner = FineTuner(learn)
    # tuner.fit_with_presizing(presizing_cycles=5,
    #                          epochs_per_cycle=[5, 10],
    #                          max_image_size=460)