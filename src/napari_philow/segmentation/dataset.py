import os.path
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage import io
from torch.utils.data import Dataset
from torchvision.transforms import functional

from data_augmentation import RandomRotation, Resize, RandomMirror, RandomCrop, Compose, RandomBrightness, RandomNoise


class PHILOWDataset(Dataset):
    def __init__(self, images_dir, labels_dir, train_csv, phase, transform, multiplier=1):
        """PHILOW Dataset. Read images, apply augmentation and preprocessing transformations.
        Args:
            images_dir (str): path to images folder
            labels_dir (str): path to labels folder
            train_csv (str): path to train csv
            phase (str): 'train' or 'val'
            transform (ImageTransform): data transfromation pipeline
            multiplier (int): How many times an image is loaded in one epoch
        """
        df = pd.read_csv(train_csv, index_col=0)
        self.names = list(df[df['train']=='checked']['filename'])
        self.phase = phase
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        img, mask = self.pull_item(index)
        return img, mask

    def pull_item(self, index):
        # read data
        img = Image.open(str(self.images[index])).convert("L")
        anno_class_img = Image.open(str(self.labels[index])).convert("L")

        # 3. 前処理を実施
        img, anno_class_img = self.transform(self.phase, img, anno_class_img)
        anno_class_img = anno_class_img.point(lambda x: x * 255)
        return functional.to_tensor(img), functional.to_tensor(anno_class_img)


class ImageTransform():
    """
    sizeでcropして512にresize
    """

    def __init__(self, size):
        self.data_transform = {
            'train': Compose([
                RandomRotation([0, 90]),
                RandomCrop(size),  # resize
                Resize(512),
                RandomMirror(),
                RandomBrightness(),
                RandomNoise()
                # transforms.ToTensor()
            ]),
            'val': Compose([
                RandomCrop(size),  # resize
                Resize(512)  # ,
                # transforms.ToTensor()
            ])
        }

    def __call__(self, phase, img, mask):
        return self.data_transform[phase](img, mask)