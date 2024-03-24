import os.path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional

from .._utils import renormalize_8bit

from .data_augmentation import RandomCrop, Compose, RandomVFlip, RandomHFlip, RondomRotateShiftScale, RandomBrightness


class PHILOWDataset(Dataset):
    def __init__(self, images_dir, labels_dir, names, phase, transform, multiplier=1):
        """PHILOW Dataset. Read images, apply augmentation and preprocessing transformations.
        Args:
            images_dir (str): path to images folder
            labels_dir (str): path to labels folder
            names (list[str]): file names
            phase (str): 'train' or 'val'
            transform (ImageTransform): data transfromation pipeline
            multiplier (int): How many times an image is loaded in one epoch
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.names = names
        self.phase = phase
        self.transform = transform
        self.multiplier = multiplier

    def __len__(self):
        return len(self.names) * self.multiplier

    def __getitem__(self, index):
        img, mask = self.pull_item(index)
        return img, mask

    def pull_item(self, index):
        # read data
        # img = Image.open(os.path.join(self.images_dir, str(self.names[index]))).convert("L")
        # img = Image.fromarray(renormalize_8bit(np.array(img)))
        index = index % len(self.names)  # Use module to wrap around the names list
        img = Image.open(os.path.join(self.images_dir, str(self.names[index])))
        if img.mode == "I":
            img = Image.fromarray(renormalize_8bit(np.array(img)))
        else:
            img = img.convert("L")
        if self.labels_dir is not None:
            mask = Image.open(os.path.join(self.labels_dir, str(self.names[index]))).convert("L")
        else:
            mask = Image.new("L", img.size, color=0)

        # 3. 前処理を実施
        img, mask = self.transform(self.phase, img, mask)
        mask = mask.point(lambda x: x * 255)  # TODO: multichannel support
        return functional.to_tensor(img), functional.to_tensor(mask)


class CristaeDataset(Dataset):
    def __init__(self, images, labels, names, phase, transform, multiplier=1):
        """PHILOW Dataset. Read images, apply augmentation and preprocessing transformations.
        Args:
            images (np.ndarray): images :: (Z, Y, X)
            labels (np.ndarray): labels :: (Z, Y, X, C)
            names (list[str]): file names
            phase (str): 'train' or 'val'
            transform (ImageTransform): data transfromation pipeline
            multiplier (int): How many times an image is loaded in one epoch
        """
        self.images = images
        self.labels = labels
        self.names = names
        self.phase = phase
        self.transform = transform
        self.multiplier = multiplier

    def __len__(self):
        return len(self.names) * self.multiplier

    def __getitem__(self, index):
        img, mask = self.pull_item(index)
        return img, mask

    def pull_item(self, index):
        index = index % len(self.names)  # Use module to wrap around the names list
        img = self.images[index]
        mask = self.labels[index]
        if img.dtype == "uint16":
            img = renormalize_8bit(img)
        img = Image.fromarray(img)
        mask = Image.fromarray(mask.astype(np.uint8))
        img, mask = self.transform(self.phase, img, mask)
        return functional.to_tensor(img), functional.to_tensor(mask)


class ImageTransform:

    def __init__(self, size):
        self.data_transform = {
            'train': Compose([
                RandomCrop(size),
                RondomRotateShiftScale([0, 90], 0.1, 0.1, [0.8, 1.2], img_size=[size, size]),
                RandomVFlip(),
                RandomHFlip()
                # transforms.ToTensor()
            ]),
            'val': Compose([
                RandomCrop(size)
                # transforms.ToTensor()
            ])
        }

    def __call__(self, phase, img, mask):
        return self.data_transform[phase](img, mask)


class CristaeImageTransform:
    def __init__(self, size):
        self.data_transform = {
            'train': Compose([
                RandomCrop(size),
                RondomRotateShiftScale([0, 90], 0.1, 0.1, [0.8, 1.2], img_size=[size, size]),
                RandomVFlip(),
                RandomHFlip(),
                RandomBrightness()
            ]),
            'val': Compose([
                RandomCrop(size)
            ])
        }

    def __call__(self, phase, img, mask):
        return self.data_transform[phase](img, mask)
