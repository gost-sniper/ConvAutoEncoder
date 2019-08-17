import argparse
import os
import time

import torch.nn as nn
import torch.utils.data
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms

import DataSet
from gyroModel import GyroModel

from Opers import findLastCheckpoint, log, prepareLoaders, test, train

parser = argparse.ArgumentParser(description='PyTorch ConAutoEncoder')
parser.add_argument('--batch_size', default=2, type=int, help='batch size')
parser.add_argument('--normal_data', default='data/normal_images', type=str, help='path of train data')
parser.add_argument('--blurry_data', default='data/blurry_images', type=str, help='path of train data')
parser.add_argument('--epoch', default=5, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
args = parser.parse_args()

data_transform = transforms.Compose([transforms.ToTensor()])
dataset = DataSet.imageDataset(args.normal_data, args.blurry_data, transform=data_transform)

train_loader, validation_loader = prepareLoaders(dataset, shuffle_dataset=True, batch_size=args.batch_size, )

save_dir = os.path.join('models', args.model)

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize the NN
model = GyroModel().to(device)

# specify loss function
criterion = nn.MSELoss()

# specify the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# setting up a learning rate scheduler 
scheduler = StepLR(optimizer, step_size=2, gamma=0.1)

# number of epochs to train the model
n_epochs = args.epoch

for epoch in range(1, n_epochs + 1):
    # monitor training loss
    train_loss = 0.0

# check if there's a checkpoint
initial_epoch = findLastCheckpoint(save_dir)

if initial_epoch > 0:
    print('resuming by loading epoch %03d' % initial_epoch)
    model = torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch))

for epoch in range(initial_epoch + 1, n_epochs + 1):
    # monitor training loss
    train_loss = 0.0

    start_time = time.time()

    # train the model #
    train_loss = train(model, epoch, train_loader, criterion, optimizer, device)

    elapsed_time = time.time() - start_time
    # print avg training statistics
    train_loss = train_loss / len(train_loader)

    log('epoch = %4d , loss = %4.4f , time = %4.2f s' % (epoch, train_loss, elapsed_time))

    torch.save(model, os.path.join(save_dir, 'model_%03d.pth' % epoch))

    # changing the learning rate for the next epoch
    scheduler.step()

    test(model, validation_loader, criterion, device)
