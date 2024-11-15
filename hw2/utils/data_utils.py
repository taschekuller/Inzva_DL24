import torch
import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from typing import Any
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class MadisonStomach(Dataset):
    '''
    Custom PyTorch Dataset class to load and preprocess images and their corresponding segmentation masks.
    
    Args:
    - data_path (str): The root directory of the dataset.
    - mode (str): The mode in which the dataset is used, either 'train' or 'test'.
    
    Attributes:
    - image_paths (list): Sorted list of file paths for images.
    - mask_paths (list): Sorted list of file paths for masks.
    - transform (Compose): Transformations to apply to the images (convert to tensor and resize).
    - mask_transform (Compose): Transformations to apply to the masks (convert to tensor and resize).
    - augment (bool): Whether to apply data augmentation (only for training mode).
    - augmentation_transforms (Compose): Augmentation transformations (horizontal flip, vertical flip, color jitter).
    '''

    def __init__(self, data_path, mode='train') -> None:
        # Load and sort image and mask file paths
        
        self.image_paths = sorted(glob.glob(os.path.join(data_path, mode, "*image.png")))
        self.mask_paths = sorted(glob.glob(os.path.join(data_path, mode, "*mask.png")))
        
        # Ensure the number of images and masks match and they are not empty!!
        assert len(self.image_paths) == len(self.mask_paths) & (len(self.image_paths) & len(self.mask_paths))  > 0, "Number of images and masks must match and be greater than 0"

        # Define transformations for images and masks: convert to tensor and resize to 256x256
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)), 
            transforms.ToTensor()
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.NEAREST), 
            transforms.ToTensor()
        ])

        # Determine if augmentation should be applied (only in 'train' mode)
        self.augment = mode == 'train'

        # Define augmentation transformations for training
        if self.augment:
            self.augmentation_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)])

        
    def __len__(self):
        # Return the total number of samples
        return len(self.image_paths)

    def __getitem__(self, index):
        '''
        Load and preprocess an image and its corresponding mask at the given index.
        
        Args:
        - index (int): Index of the sample to fetch.
        
        Returns:
        - img (Tensor): Transformed image tensor.
        - mask (Tensor): Transformed mask tensor.
        '''
        # Load the image and mask using OpenCV (image in grayscale, mask with unchanged properties)
        img = cv2.imread(self.image_paths[index], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.mask_paths[index], cv2.IMREAD_UNCHANGED)
        
        # Apply transformations to the image and mask
        img = self.transform(Image.fromarray(img))
        mask = self.mask_transform(Image.fromarray(mask))

        # Apply data augmentation if enabled
        if self.augment:
            # Set random seed to ensure consistency between image and mask transformations
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed) #TODO Check this
            # torch.manual_seed(42)

        return img, mask
