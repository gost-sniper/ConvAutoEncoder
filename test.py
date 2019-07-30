import os

import torch
from torchvision import transforms

import DataSet
from Opers import findLastCheckpoint, prepareLoaders

save_dir = os.path.join('models', 'ConAutoEncoder')

Last_checkpoint = findLastCheckpoint(save_dir)

model_location = os.path.join(save_dir, 'model_%03d.pth' % Last_checkpoint)

data_transform = transforms.Compose([transforms.ToTensor()])
dataset = DataSet.imageDataset('data/normal_images', 'data/normal_images', transform=data_transform)

train_loader, validation_loader = prepareLoaders(dataset, shuffle_dataset=True, batch_size=2, )


model = torch.load(model_location)
