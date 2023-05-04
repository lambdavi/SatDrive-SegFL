import os
from typing import Any
import numpy as np
from PIL import Image
from torch import from_numpy
from torchvision.datasets import VisionDataset
import datasets.ss_transforms as tr
from torchvision.utils import save_image

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class_map = {
   1: 13,  # ego_vehicle : vehicle
   7: 0,   # road
   8: 1,   # sidewalk
   11: 2,  # building
   12: 3,  # wall
   13: 4,  # fence
   17: 5,  # pole
   18: 5,  # poleGroup: pole
   19: 6,  # traffic light
   20: 7,  # traffic sign
   21: 8,  # vegetation
   22: 9,  # terrain
   23: 10,  # sky
   24: 11,  # person
   25: 12,  # rider
   26: 13,  # car : vehicle
   27: 13,  # truck : vehicle
   28: 13,  # bus : vehicle
   32: 14,  # motorcycle
   33: 15,  # bicycle
}

class GTA5Dataset(VisionDataset):

    def __init__(self,
                 root: str,
                 list_samples: list[str],
                 transform: tr.Compose = None,
                 client_name: str = None):
        super().__init__(root=root, transform=transform, target_transform=None)
        self.list_samples = list_samples
        self.client_name = client_name
        self.target_transform = self.get_mapping()
        
    @staticmethod
    def get_mapping():
        mapping = np.zeros((256,), dtype=np.int64) + 255
        for gta_idx, data_idx in class_map.items():
            mapping[gta_idx] = data_idx
        return lambda x: from_numpy(mapping[x])

    def __getitem__(self, index: int) -> Any:
        image = Image.open(os.path.join(self.root, self.list_samples[index]+".png")).convert("RGB")
        label = Image.open(os.path.join(self.root, self.list_samples[index]+".png"))
        if self.transform: # != None
            image, label = self.transform(image, label)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def __len__(self) -> int:
        return len(self.list_samples)
