#%%

import fastbook
fastbook.setup_book()

#%%

from fastbook import *
from fastai.vision.widgets import *
from fastai.callback.tensorboard import *

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
    return [img for img in get_image_files(path) if sort_image_by_dimensions(img, image_size)]

#%%

def get_y(r): return parent_label(r).split(" ")

#%%

def get_dls(bs, size):
    dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                        splitter=RandomSplitter(seed=42),
                        get_items=correct_sized_images,
                        get_y=get_y,
                        item_tfms=Resize(image_size),
                        batch_tfms=aug_transforms(size=size, min_scale=0.75))

    return dblock.dataloaders(path, bs=bs, num_workers=4)

#%%

batch_size = 64
dls = get_dls(batch_size, 64)

#%%

dls.show_batch(max_n=4, nrows=1)

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

from fastai.callback.fp16 import *
learn = cnn_learner(dls,
                    resnet152,
                    metrics=partial(accuracy_multi, thresh=0.2),
                    loss_func=smooth_loss_v2(pos_weight=pos_weight),
                    cbs=MixUp)

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
        with self.learner.no_bar():
            self.learner.fine_tune(epochs)

    def get_first_index_else_last(self, list_obj: list, idx: int, total_iters: int):
        item = list_obj[0]

        if idx == total_iters - 1:
            item = list_obj[-1]

        return list_obj[idx] if idx < len(list_obj) - 1 else item

#%%

tuner = FineTuner(learn)
tuner.fit_with_presizing(presizing_cycles=5,
                         epochs_per_cycle=[5, 10],
                         max_image_size=460)

#%%

# with learn.no_bar():
#     learn.fine_tune(5, freeze_epochs=3)

#%%

# learn.dls = get_dls(batch_size, 128)

#%%

# with learn.no_bar():
#     learn.fine_tune(5)

#%%

# learn.dls = get_dls(batch_size, 256)

#%%

# with learn.no_bar():
#     learn.fine_tune(10)

#%%

learn.loss_func

#%%

learn.recorder.plot_loss()

#%%

x,y = dls.one_batch()

#%%

import random as rnd

preds,t = learn.get_preds(dl=[(x,y)])
rand_idx = rnd.randint(0, len(preds))-1
print(preds[rand_idx], t[rand_idx])

#%%

preds[0].sum()

#%%

learn.model(x).shape

#%%

model_name = "resnet18"
time_now = datetime.datetime.now().strftime(format='%H%M%S')

export_path = Path('../models/exported')

model_filename = export_path/f"{time_now}_{model_name}_B{batch_size}S{image_size}.pkl"
learn.export(fname=model_filename)

#%%

learn.dls.vocab
