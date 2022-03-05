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


class BSDS300Dataset(Dataset):
    def __init__(self, root='./BSDS300', patch_size=32, split='train', use_patches=True):
        files = sorted(glob(os.path.join(root, 'images', split, '*')))

        self.use_patches = use_patches
        self.images = self.load_images(files)
        self.patches = self.patchify(self.images, patch_size)
        self.mean = torch.mean(self.patches)
        self.std = torch.std(self.patches)

    def load_images(self, files):
        out = []
        for fname in files:
            img = skimage.io.imread(fname).astype(np.float32) / 255.
            if img.shape[0] > img.shape[1]:
                img = img.transpose(1, 0, 2)
            img = resize(img, (32,32), anti_aliasing=True)
            img = img.transpose(2, 0, 1)
            out.append(torch.from_numpy(img))
        return torch.stack(out)

    def patchify(self, img_array, patch_size):
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
