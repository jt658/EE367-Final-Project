"""
This script provides a Dataset object for the SIDD Dataset (https://www.eecs.yorku.ca/~kamel/sidd/). 

The SIDDDataset object can return whole images, or it can break images into uniform patches and return
individual patches instead.
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

#matplotlib.rcParams['figure.raise_window'] = False

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SIDDDataset(Dataset):
    def __init__(self, root='./SIDD/data/*/', patch_size=32, use_patches=True):
        self.gt_files = sorted(glob(os.path.join(root, 'GT*')))
        self.noisy_files = sorted(glob(os.path.join(root, 'NOISY*')))
        self.use_patches = use_patches
        self.patch_size = patch_size
        
        # pytorch dataloader requires a __len__ attribute for this dataset
        # because images in the dataset decompose into different numbers of 
        # patches, there isn't an easy way to signal to the dataloader the 
        # total number of patches we get from all images in our dataset (without,
        # of course, reading in each image from disk, patchifying, and summing up
        # the number of patches). Instead, I'm just estimating the total number of
        # patches.
        if self.use_patches:
            example_img, _ = self.load_image(self.gt_files[-1])
            self.patches_per_image = self.patchify(torch.unsqueeze(example_img, dim=0), self.patch_size).shape[0]
            self.len = self.patches_per_image * len(self.gt_files)    # <-- just a heuristic, not necessarily accurate
            print(f"patches per image = {self.patches_per_image}")
        else:
            self.len = len(self.gt_files)
            
        # for faster memory accesses / bypassing disk reads
        # cache patches that you have read into memory so that you don't
        # have to read them in again
        self.loaded_img_idx = None
        self.loaded_noisy_patches = None
        self.loaded_img_patches = None

    def load_image(self, fname, reshape=False):
        """
        Loads all images in dataset into a tensor.
        """

        img = skimage.io.imread(fname).astype(np.float32) / 255.
        if img.shape[0] > img.shape[1]:
            img = img.transpose(1, 0, 2)
        H, W = img.shape[0], img.shape[1]
        if reshape:
          img = resize(img, (img.shape[0] // 4, img.shape[1] // 4), anti_aliasing=True)
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img), (H, W)
        
    def patchify(self, img_array, patch_size):
        """
        Divides images into patches of uniform size. 
        """

        # create patches from image array of size (N_images, 3, rows, cols)
        patches = img_array.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.reshape(patches.shape[0], 3, -1, patch_size, patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).reshape(-1, 3, patch_size, patch_size)
        return patches

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        if self.use_patches:
            img_idx = int(idx / self.patches_per_image)

            if self.loaded_img_idx != img_idx:
                # need to update image cache
                self.loaded_img_idx = img_idx
                gt_image, _ = self.load_image(self.gt_files[img_idx])
                noisy_image, _ = self.load_image(self.noisy_files[img_idx])
                
                self.loaded_gt_patches = self.patchify(torch.unsqueeze(gt_image, dim=0), self.patch_size)
                self.loaded_noisy_patches = self.patchify(torch.unsqueeze(noisy_image, dim=0), self.patch_size)

            num_patches_per_image = self.loaded_gt_patches.shape[0]
            patch_idx = idx % num_patches_per_image        # I know this is super jank 

            return (self.loaded_gt_patches[patch_idx], self.loaded_noisy_patches[patch_idx])
        else:
            gt_image, img_dim_gt = self.load_image(self.gt_files[idx], reshape=True)
            noisy_image, img_dim_noisy = self.load_image(self.noisy_files[idx], reshape=True)
            return (gt_image, noisy_image), img_dim_gt
