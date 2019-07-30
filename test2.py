import os

import torch
from torchvision import transforms

import DataSet
from Opers import findLastCheckpoint, prepareLoaders
from model2 import GyroModel

data_transform = transforms.Compose([transforms.ToTensor()])
dataset = DataSet.imageDataset('data/normal_images', 'data/normal_images', transform=data_transform)

train_loader, validation_loader = prepareLoaders(dataset, shuffle_dataset=True, batch_size=2, )

data = next(iter(train_loader))

blurry, normal = data['blurry'], data['normal']

model = GyroModel(3)

y = model(blurry)
