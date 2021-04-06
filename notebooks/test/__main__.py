from fastai.vision.all import *
from torchvision import datasets, transforms

import fastbook
fastbook.setup_book()

from fastbook import *
from fastai.vision.widgets import *

class Net(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(1, 32, 3, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1), nn.MaxPool2d(2), nn.Dropout2d(0.25),
            Flatten(), nn.Linear(9216, 128), nn.ReLU(), nn.Dropout2d(0.5),
            nn.Linear(128, 10), nn.LogSoftmax(dim=1))


batch_size, test_batch_size = 256, 512
epochs, lr = 1, 1e-2

kwargs = {'num_workers': 0}
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])
train_loader = DataLoader(
    datasets.MNIST('../data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = DataLoader(
    datasets.MNIST('../data', train=False, transform=transform),
    batch_size=test_batch_size, shuffle=True, **kwargs)

path = untar_data(URLs.MNIST)

dblock = DataBlock(blocks=(ImageBlock(cls=PILImageBW), MultiCategoryBlock),
                   splitter=RandomSplitter(seed=42),
                   get_items=get_image_files,
                   get_y=parent_label)

dls = dblock.dataloaders(path, bs=64, num_workers=0).cuda()

if __name__ == '__main__':
    # print(path.ls())
    # data = DataLoaders(train_loader, test_loader).cuda()
    # dls = ImageDataLoaders.from_folder(path)
    learn = Learner(dls, Net(), metrics=accuracy)
    # learn = cnn_learner(dls, resnet18, pretrained=False, metrics=accuracy)
    learn.fit_one_cycle(epochs, lr)
