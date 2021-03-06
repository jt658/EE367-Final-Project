"""
This script provides a Dataset object for the BSDS300 dataset (https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/). 

The Dataset object can sample entire images from the BSDS300 dataset, or it can return smaller patches of an image.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from glob import glob
import os
import skimage.io
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

# set random seeds
torch.manual_seed(1)
torch.use_deterministic_algorithms(True)

class BSDS300Dataset(Dataset):
    def __init__(self, root='./BSDS300', patch_size=32, split='train', use_patches=True, reshape=False):
        """
        :param root (str): directory containing images from BSDS300 dataset
        :param patch_size (int): the dimension of a patch, which we subsample from images
        :param split (str): 'train' or 'test', determines which group of images we will sample from
        :param use_patches (boolean): True if object will return patches, False if object will return 
            entire images
        :param reshape (boolean): True if images will be downsampled
        """
        files = sorted(glob(os.path.join(root, 'images', split, '*')))

        self.use_patches = use_patches
        self.images = self.load_images(files, reshape)
        self.patches = self.patchify(self.images, patch_size)
        self.mean = torch.mean(self.patches)
        self.std = torch.std(self.patches)

    def load_images(self, files, reshape=False):
        """
        Loads all images in dataset into a tensor. 
        """

        out = []
        for fname in files:
            img = skimage.io.imread(fname)
            if img.shape[0] > img.shape[1]:
                img = img.transpose(1, 0, 2)
            if reshape: 
                img = img.astype(np.float32) / 255.
                img = resize(img, (img.shape[0] // 8, img.shape[1] // 8), anti_aliasing=True)
                img = img.transpose(2, 0, 1) # Move astype and stuff after to line 35 if doing resizing
            else:
                img = img.transpose(2, 0, 1).astype(np.float32) / 255.
            out.append(torch.from_numpy(img))
        return torch.stack(out)

    def patchify(self, img_array, patch_size):
        """
        Divides images into smaller patches of dimension patch_size.
        """
        # create patches from image array of size (N_images, 3, rows, cols)
        patches = img_array.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.reshape(patches.shape[0], 3, -1, patch_size, patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).reshape(-1, 3, patch_size, patch_size)
        return patches

    def __len__(self):
        if self.use_patches:
            return self.patches.shape[0]
        else:
            return self.images.shape[0]

    def __getitem__(self, idx):
        if self.use_patches:
            return self.patches[idx]
        else:
            return self.images[idx]
