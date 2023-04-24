import os
from typing import Any
import numpy as np
from PIL import Image
from torch import from_numpy
from torchvision.datasets import VisionDataset
import datasets.ss_transforms as tr
from torchvision.utils import save_image

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class_eval = [255, 2, 4, 255, 11, 5, 0, 0, 1, 8, 13, 3, 7, 6, 255, 255, 15, 14, 12, 9, 10]
IMG_DIR = "images"
LAB_DIR = "labels"

class IDDADataset(VisionDataset):

    def __init__(self,
                 root: str,
                 list_samples: [str],
                 transform: tr.Compose = None,
                 client_name: str = None):
        super().__init__(root=root, transform=transform, target_transform=None)
        self.list_samples = list_samples
        self.client_name = client_name
        self.target_transform = self.get_mapping()
    @staticmethod
    def get_mapping():
        classes = class_eval
        mapping = np.zeros((256,), dtype=np.int64) + 255
        for i, cl in enumerate(classes):
            mapping[i] = cl
        return lambda x: from_numpy(mapping[x])

    def __getitem__(self, index: int) -> Any:
        image = Image.open(os.path.join(self.root,IMG_DIR, self.list_samples[index]+".jpg")).convert("RGB")
        label = Image.open(os.path.join(self.root, LAB_DIR, self.list_samples[index]+".png"))
        if self.transform: # != None
            image, label = self.transform(image, label)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def __len__(self) -> int:
        return len(self.list_samples)
