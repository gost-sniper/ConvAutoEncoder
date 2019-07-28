from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import glob
import os

import Opers


class imageDataset(Dataset):

    def __init__(self, normal_dir, blurry_dir, transform=None):
        working_dir = os.path.dirname(os.path.realpath(__file__))

        self.path_normal_dir = os.path.join(working_dir, normal_dir)
        self.path_blurry_dir = os.path.join(working_dir, blurry_dir)

        self.blurry_images = [x.replace(self.path_blurry_dir, '') for x in glob.glob(self.path_blurry_dir + '/*.jpg')]
        self.normal_images = [x.replace(self.path_normal_dir, '') for x in glob.glob(self.path_normal_dir + '/*.jpg')]

        self.transform = transform

        if self.blurry_images not in self.normal_images and len(self.blurry_images) != len(self.normal_images):
            raise Exception('mismatch between the normal images and the blurry ones')

    def __len__(self):
        return len(self.normal_images)

    def __getitem__(self, idx):
        normal_image = io.imread(self.path_normal_dir + '/' + self.normal_images[idx])
        blurry_image = io.imread(self.path_blurry_dir + '/' + self.normal_images[idx])

        sample = {'normal': normal_image, 'blurry': blurry_image}

        if self.transform:
            n, b = self.transform(sample['normal']), self.transform(sample['blurry'])
            sample = {'normal': n, 'blurry': b}

        return sample
