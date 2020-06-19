from __future__ import print_function

import torch.utils.data as data
import os
import glob
from PIL import Image
from utils import preprocess


class OurDataset(data.Dataset):
    CLASSES = [
        "background",
        "road",
        "side-walk",
        "people",
        "car",
        "building",
        "bridge",
        "median",
        "sky",
        "plant",
        "inner-car",
    ]

    def __init__(
        self, root, train=True, transform=None, target_transform=None, crop_size=None,
    ):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.crop_size = crop_size

        jpgs = glob.glob("{}/*.jpg".format(self.root))
        print("total has {} images in ourdataset({}).".format(len(jpgs), self.root))
        if self.train:
            jpgs = jpgs[:-20]
        else:
            jpgs = jpgs[-20:]

        self.images = []
        self.masks = []
        for jpg in jpgs:
            self.images.append(jpg)
            self.masks.append(jpg.replace(".jpg", ".png"))

    def __getitem__(self, index):
        _img = Image.open(self.images[index]).convert("RGB")
        _target = Image.open(self.masks[index])

        _img, _target = preprocess(
            _img,
            _target,
            flip=True if self.train else False,
            scale=(0.5, 2.0) if self.train else None,
            crop=(self.crop_size, self.crop_size),
            is_train=self.train,
        )

        if self.transform is not None:
            _img = self.transform(_img)

        if self.target_transform is not None:
            _target = self.target_transform(_target)

        return _img, _target

    def __len__(self):
        return len(self.images)
