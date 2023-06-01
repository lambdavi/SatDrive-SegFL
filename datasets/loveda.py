import os
from typing import Any
import numpy as np
from PIL import Image
from torch import from_numpy
from torchvision.datasets import VisionDataset
import datasets.ss_transforms as tr

class_eval = [255, 1, 2, 3, 4, 5, 6]
IMG_DIR = "images"
LAB_DIR = "labels"

class LoveDADataset(VisionDataset):
    def __init__(self,
                 root: str,
                 list_samples: list[str],
                 transform: tr.Compose = None,
                 client_name: str = None):
        super().__init__(root=root, transform=transform, target_transform=None)
        self.list_samples = list_samples
        self.client_name = client_name
        self.target_transform = self.get_mapping()
        self.style_tf_fn = None
        self.return_unprocessed_image = False

    def set_style_tf_fn(self, style_tf_fn):
        self.style_tf_fn = style_tf_fn

    def reset_style_tf_fn(self):
        self.style_tf_fn = None

    @staticmethod
    def get_mapping():
        classes = class_eval
        mapping = np.zeros((256,), dtype=np.int64) + 255
        for i, cl in enumerate(classes):
            mapping[i] = cl
        return lambda x: from_numpy(mapping[x])

    def __getitem__(self, index: int) -> Any:
        image = Image.open(os.path.join(self.root,IMG_DIR, self.list_samples[index]+".png")).convert("RGB")
        label = Image.open(os.path.join(self.root, LAB_DIR, self.list_samples[index]+".png"))

        if self.return_unprocessed_image:
            return image
        
        if self.style_tf_fn is not None:
            image = self.style_tf_fn(image)

        if self.transform is not None: 
            if isinstance(self.transform, list):
                image = self.transform[0](image)
                image, label = self.transform[1](image, label)
            else:
                image, label = self.transform(image, label)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def __len__(self) -> int:
        return len(self.list_samples)
