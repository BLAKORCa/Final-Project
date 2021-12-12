import torch
import torch.nn as nn
from torchvision import models 
from torchvision.models import mobilenet_v2, vgg16

class cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(#
            nn.Conv2d(3,10,kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(10,15,kernel_size=3,stride=1,padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2,2), #

            nn.Conv2d(15, 20, kernel_size=3,stride=2,padding=0),
            nn.ReLU(),
            nn.Conv2d(20,20,kernel_size=3,stride=1,padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(20,25,kernel_size=3,stride=1,padding=0),
            nn.ReLU(),
            nn.Conv2d(25,25,kernel_size=3,stride=1,padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Flatten(),
            nn.Linear(400,128),
            nn.Dropout(0.25),
            nn.Linear(128,100),
            nn.LogSoftmax(1)
        )

    def forward(self,x):
        return self.feature(x)


class MobileNetv2(nn.Module):
  def __init__(self, output_size=100):
    super().__init__()
    self.mnet = mobilenet_v2(pretrained=True)

    self.mnet.classifier = nn.Sequential(
        nn.Linear(1280, output_size),
        nn.LogSoftmax(1)
    )

  def forward(self, x):
    return self.mnet(x)

class SpinalNet_ResNet(nn.Module):
    
    def __init__(self):
        super(SpinalNet_ResNet, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.linear = nn.Linear(self.model.fc.in_features, 100)

    def forward(self, x):
        x = self.model(x)
        x = self.linear(x)
        return x

class Vgg16(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.vnet = vgg16(pretrained=True)
        self.vnet.classifier = nn.Sequential(
        nn.Linear(25088, output_size))
    def forward(self, x):
        return self.vnet(x)