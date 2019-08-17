import datetime
import glob
import os
import re

import torch
import numpy as np
from skimage import transform
from torch.utils.data.sampler import SubsetRandomSampler



class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        normal, blurry = sample['normal'], sample['blurry']

        h, w = normal.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(normal, (new_h, new_w))
        blurry = transform.resize(blurry, (new_h, new_w))

        return {'normal': img, 'blurry': blurry}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        normal, blurry = sample['normal'], sample['blurry']

        h, w = normal.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        normal = normal[top: top + new_h,
                 left: left + new_w]
        blurry = blurry[top: top + new_h,
                 left: left + new_w]

        return {'normal': normal, 'blurry': blurry}


class ToTensor(object):
    """Convert nd-arrays in sample to Tensors."""

    def __call__(self, sample):
        normal, blurry = sample['normal'], sample['blurry']

        # swap color axis because
        # numpy normal: H x W x C
        # torch normal: C X H X W
        normal = normal.transpose((2, 0, 1))
        return {'normal': torch.from_numpy(normal),
                'blurry': torch.from_numpy(blurry)}


def findLastCheckpoint(save_dir):
    """Search for the latest checkpoint
        
    Args:
        save_dir (str): path to the directory where model's
            checkpoints are saved
    """
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def log(*args, **kwargs):
    """Print important logs"""
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


def prepareLoaders(dataset, shuffle_dataset=True, batch_size=4, random_seed=42, validation_split=.15):
    """Prepare training and validation loaders

    Args:
        dataset (torch.utils.data.Dataset object): the whole dataset that we want 
            to split into training-set and validation-set.
        shuffle_dataset (boolean) : specify whether we want shuffle the 
            data before spliting.(Default True)
        random_seed (int) : seed for the random shuffle. (Default 42)
        validation_split (float): the split ratio fortraining-set and 
            validation-set it should be between 0 and 1. (Default 0.15) 
    """
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # building dataset slicers
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    # building dataset loaders
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)
    return train_loader, validation_loader



def train(model, epoch, train_loader, criterion, optimizer, device):
    """One pass through the trainning set

    Args:
        model (torch.nn.Module): the model you want to to train
        epoch (int): the number of the current epoch
        train_loader (torch.utils.data.DataLoader): training-set dataloader
        criterion (callable): the loss function used to evaluate the model
        optimizer (callable): the optimizer used to minimise the loss function
        device (torch.device): device used for calculations 'CPU' or 'GPU'

    """
    train_loss = .0
    model.train()
    for batch_idx, data in enumerate(train_loader):
        blurry, normal = data['blurry'].to(device), data['normal'].to(device)

        optimizer.zero_grad()
        output = model(blurry)
        loss = criterion(output, normal)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch_idx % 25 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(blurry), len(train_loader),
                100. * batch_idx / len(train_loader), loss.item()))

    return train_loss


def test(model, validation_loader, criterion, device):
    """evaluate the current state of the model

    Args:
        model (torch.nn.Module): the model you want to to train
        validation_loader (torch.utils.data.DataLoader): validation-set dataloader
        criterion (callable): the loss function used to evaluate the model
        device (torch.device): device used for calculations 'CPU' or 'GPU'

    """
    with torch.no_grad():
        model.eval()
        test_loss = 0
        for data in validation_loader:
            blurry, normal = data['blurry'].to(device), data['normal'].to(device)
            output = model(blurry)
            # sum up batch loss
            test_loss += criterion(output, normal, ).item()

        test_loss /= len(validation_loader.dataset)
        print('\nTest set: Average loss: {:.4f}\n'
              .format(test_loss))