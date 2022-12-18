import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from .model_utils.unet_parts import *


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class CNN(BaseModel):
    def __init__(self, output_channels=3):
        super().__init__()
        self.kernel_size = 5
        self.padding = ( 1 * (1-1)- 1 + 1*(self.kernel_size - 1))//2 + 1

        self.conv1 = nn.Conv2d(3, 16, self.kernel_size, padding=self.padding)
        self.conv2 = nn.Conv2d(16, 64, self.kernel_size, padding=self.padding)
        self.conv3 = nn.Conv2d(64, output_channels, self.kernel_size, padding=self.padding)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x


class CNN6(BaseModel):
    def __init__(self, output_channels=3):
        super().__init__()
        self.kernel_size = 5
        self.padding = ( 1 * (1-1)- 1 + 1*(self.kernel_size - 1))//2 + 1

        self.conv1 = nn.Conv2d(3, 64, self.kernel_size, padding=self.padding)
        self.conv2 = nn.Conv2d(64, 128, self.kernel_size, padding=self.padding)
        self.conv3 = nn.Conv2d(128, 256, self.kernel_size, padding=self.padding)
        self.conv4 = nn.Conv2d(256, 128, self.kernel_size, padding=self.padding)
        self.conv5 = nn.Conv2d(128, 64, self.kernel_size, padding=self.padding)
        self.conv6 = nn.Conv2d(64, output_channels, self.kernel_size, padding=self.padding)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        return x


class CNN_photo_finishing(BaseModel):
    def __init__(self, output_channels=3):
        super().__init__()
        self.kernel_size = 3
        self.padding = ( 1 * (1-1)- 1 + 1*(self.kernel_size - 1))//2 + 1

        self.conv1 = nn.Conv2d( 3, 64, self.kernel_size, padding=self.padding)
        self.conv2 = nn.Conv2d(64, 64, self.kernel_size, padding=self.padding)
        self.conv3 = nn.Conv2d(64, 64, self.kernel_size, padding=self.padding)
        self.conv4 = nn.Conv2d(64, 32, self.kernel_size, padding=self.padding)
        self.conv5 = nn.Conv2d(32,  output_channels, self.kernel_size, padding=self.padding)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = F.leaky_relu(self.conv5(x), 0.2)
        return x


class CNN_photo_finishing_single(BaseModel):
    def __init__(self, output_channels=1):
        super().__init__()
        self.kernel_size = 3
        self.padding = ( 1 * (1-1)- 1 + 1*(self.kernel_size - 1))//2 + 1

        self.conv1 = nn.Conv2d( 3, 64, self.kernel_size, padding=self.padding)
        self.conv2 = nn.Conv2d(64, 64, self.kernel_size, padding=self.padding)
        self.conv3 = nn.Conv2d(64, 64, self.kernel_size, padding=self.padding)
        self.conv4 = nn.Conv2d(64, 32, self.kernel_size, padding=self.padding)
        self.conv5 = nn.Conv2d(32,  output_channels, self.kernel_size, padding=self.padding)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = F.leaky_relu(self.conv5(x), 0.2)
        return x

class resnet(BaseModel):
    def __init__(self, output_channels=3):
        super().__init__()
        self.kernel_size = 3
        self.padding = ( 1 * (1-1)- 1 + 1*(self.kernel_size - 1))//2 + 1

        self.conv1 = nn.Conv2d( 3, 64, self.kernel_size, padding=self.padding)
        self.conv2 = nn.Conv2d(64, 64, self.kernel_size, padding=self.padding)
        self.conv3 = nn.Conv2d(64, 64, self.kernel_size, padding=self.padding)
        self.conv4 = nn.Conv2d(64, 32, self.kernel_size, padding=self.padding)
        self.conv5 = nn.Conv2d(32,  output_channels, self.kernel_size, padding=self.padding)

    def forward(self, x):
        
        out = F.leaky_relu(self.conv1(x), 0.1)
        x1 = out
        out = F.leaky_relu(self.conv2(x1 + out), 0.1)
        x2 = out
        out = F.leaky_relu(self.conv3(x2 + out), 0.1)
        x3 = out
        out = F.leaky_relu(self.conv4(x3 + out), 0.1)
        x4 = out
        out = F.leaky_relu(self.conv5(x4), 0.1)
        return out


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits