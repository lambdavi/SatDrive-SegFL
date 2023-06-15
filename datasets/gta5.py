import os
from typing import Any
import numpy as np
from PIL import Image
from torch import from_numpy
from torchvision.datasets import VisionDataset
import datasets.ss_transforms as tr
from torchvision.utils import save_image


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

IMAGES_DIR = "images"
LABELS_DIR = "labels"

class GTA5Dataset(VisionDataset):
    """
    Custom dataset class for GTA5 dataset.

    Args:
        root (str): Root directory of the dataset.
        list_samples (list[str]): List of sample names.
        transform (tr.Compose, optional): Transformations to apply to the samples. Defaults to None.
        client_name (str, optional): Name of the client. Defaults to None.
    """
    def __init__(self,
                 root: str,
                 list_samples: list[str],
                 transform: tr.Compose = None,
                 client_name: str = None):
        super().__init__(root=root, transform=transform, target_transform=None)
        self.list_samples = list_samples
        self.client_name = client_name
        self.target_transform = self.get_mapping()
        # Setup for style transfer
        self.style_tf_fn = None
        self.return_unprocessed_image = False
    
    def set_style_tf_fn(self, style_tf_fn):
        """
        Sets the style transfer function.

        Args:
            style_tf_fn: Style transfer function.
        """
        self.style_tf_fn = style_tf_fn

    def reset_style_tf_fn(self):
        """Resets the style transfer function."""
        self.style_tf_fn = None

    @staticmethod
    def get_mapping():
        """
        Generates a mapping function for target labels.

        Returns:
            The mapping function.
        """
        mapping = np.zeros((256,), dtype=np.int64) + 255
        for gta_idx, data_idx in class_map.items():
            mapping[gta_idx] = data_idx
        return lambda x: from_numpy(mapping[x])

    def __getitem__(self, index: int) -> Any:
        """
        Retrieves an item from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            tuple: Tuple containing the image and label.
        """
        image = Image.open(os.path.join(self.root, IMAGES_DIR, self.list_samples[index])).convert("RGB")
        label = Image.open(os.path.join(self.root, LABELS_DIR, self.list_samples[index]))

        if self.return_unprocessed_image:
            return image
        
        # Apply style transfer
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
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.list_samples)
