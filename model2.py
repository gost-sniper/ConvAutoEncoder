import torch.nn as nn
import torch.nn.functional as F


class GyroModel(nn.Module):
    """docstring for gyroModel"""

    def __init__(self, n_channels):
        super(GyroModel, self).__init__()
        kernel_size = 3
        padding = 1
        # ------------------------------------------------------ #
        #                   initial channels                     #
        # ------------------------------------------------------ #
        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=64,
                               kernel_size=kernel_size, padding=padding, bias=False, stride=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64,
                               kernel_size=kernel_size, padding=padding, bias=False, stride=1)

        self.mp1 = nn.MaxPool2d(kernel_size=2)
        # ------------------------------------------------------ #
        #                      64 channels                       #
        # ------------------------------------------------------ #
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=kernel_size, padding=padding, bias=False, stride=1)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=kernel_size, padding=padding, bias=False, stride=1)

        self.mp2 = nn.MaxPool2d(kernel_size=2)
        # ------------------------------------------------------ #
        #                     128 channels                       #
        # ------------------------------------------------------ #
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=kernel_size, padding=padding, bias=False, stride=1)

        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256,
                               kernel_size=kernel_size, padding=padding, bias=False, stride=1)

        self.mp3 = nn.MaxPool2d(kernel_size=2)

        # ------------------------------------------------------ #
        #                     256 channels                       #
        # ------------------------------------------------------ #
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=512,
                               kernel_size=kernel_size, padding=padding, bias=False, stride=1)

        self.conv8 = nn.Conv2d(in_channels=512, out_channels=512,
                               kernel_size=kernel_size, padding=padding, bias=False, stride=1)

        self.mp4 = nn.MaxPool2d(kernel_size=2)
        # ------------------------------------------------------ #
        #                     512 channels                       #
        # ------------------------------------------------------ #
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=1024,
                               kernel_size=kernel_size, padding=padding, bias=False, stride=1)

        self.conv10 = nn.Conv2d(in_channels=1024, out_channels=1024,
                                kernel_size=kernel_size, padding=padding, bias=False, stride=1)

        # ------------------------------------------------------ #
        #                     512 channels                       #
        # ------------------------------------------------------ #

        self.upconv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2,
                                          stride=2, padding=1)

        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512,
                                kernel_size=kernel_size, padding=padding, bias=False, stride=1)
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512,
                                kernel_size=kernel_size, padding=padding, bias=False, stride=1)
        # ------------------------------------------------------ #
        #                     256 channels                       #
        # ------------------------------------------------------ #
        self.upconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=1)

        self.conv13 = nn.Conv2d(in_channels=256, out_channels=256,
                                kernel_size=kernel_size, padding=padding, bias=False, stride=1)
        self.conv14 = nn.Conv2d(in_channels=256, out_channels=256,
                                kernel_size=kernel_size, padding=padding, bias=False, stride=1)
        # ------------------------------------------------------ #
        #                     128 channels                       #
        # ------------------------------------------------------ #
        self.upconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=1)

        self.conv15 = nn.Conv2d(in_channels=128, out_channels=128,
                                kernel_size=kernel_size, padding=padding, bias=False, stride=1)
        self.conv16 = nn.Conv2d(in_channels=128, out_channels=128,
                                kernel_size=kernel_size, padding=padding, bias=False, stride=1)

        # ------------------------------------------------------ #
        #                     64 channels                       #
        # ------------------------------------------------------ #
        self.upconv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=1)

        self.conv17 = nn.Conv2d(in_channels=64, out_channels=64,
                                kernel_size=kernel_size, padding=padding, bias=False, stride=1)
        self.conv18 = nn.Conv2d(in_channels=64, out_channels=64,
                                kernel_size=kernel_size, padding=padding, bias=False, stride=1)

        # ------------------------------------------------------ #
        #                     3 channels                       #
        # ------------------------------------------------------ #
        self.conv20 = nn.Conv2d(in_channels=64, out_channels=3,
                                kernel_size=1, padding=0, bias=False)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        res1 = x
        x = self.mp1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        res2 = x
        x = self.mp2(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        res3 = x
        x = self.mp3(x)

        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        res4 = x
        x = self.mp4(x)

        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))

        x = self.upconv1(x)
        x = nn.ConstantPad2d((1, 1, 2, 1), 0)(x)
        x = x.add(res4)
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))

        x = self.upconv2(x)
        x = nn.ConstantPad2d((1, 1, 1, 1), 0)(x)
        x = x.add(res3)
        x = F.relu(self.conv13(x))
        x = F.relu(self.conv14(x))

        x = self.upconv3(x)
        x = nn.ConstantPad2d((1, 1, 1, 1), 0)(x)
        x = x.add(res2)
        x = F.relu(self.conv15(x))
        x = F.relu(self.conv16(x))

        x = self.upconv4(x)
        x = nn.ConstantPad2d((1, 1, 1, 1), 0)(x)
        x = x.add(res1)
        x = F.relu(self.conv17(x))
        x = F.relu(self.conv18(x))

        x = self.conv20(x)

        return x
