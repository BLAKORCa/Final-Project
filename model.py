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
            nn.Linear(128,100)
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
        
        model_ft = models.wide_resnet101_2(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        self.half_in_size = round(num_ftrs / 2)
        layer_width = 540
        Num_class = 100
        
        self.fc_spinal_layer1 = nn.Sequential(
            #nn.Dropout(p = 0.5),
            nn.Linear(self.half_in_size, layer_width),
            #nn.BatchNorm1d(layer_width),
            nn.ReLU(inplace=True),)
        self.fc_spinal_layer2 = nn.Sequential(
            #nn.Dropout(p = 0.5),
            nn.Linear(self.half_in_size+layer_width, layer_width),
            #nn.BatchNorm1d(layer_width),
            nn.ReLU(inplace=True),)
        self.fc_spinal_layer3 = nn.Sequential(
            #nn.Dropout(p = 0.5),
            nn.Linear(self.half_in_size+layer_width, layer_width),
            #nn.BatchNorm1d(layer_width),
            nn.ReLU(inplace=True),)
        self.fc_spinal_layer4 = nn.Sequential(
            #nn.Dropout(p = 0.5),
            nn.Linear(self.half_in_size+layer_width, layer_width),
            #nn.BatchNorm1d(layer_width),
            nn.ReLU(inplace=True),)
        self.fc_out = nn.Sequential(
            #nn.Dropout(p = 0.5),
            nn.Linear(layer_width*4, Num_class),)

    def forward(self, x):
        x1 = self.fc_spinal_layer1(x[:, 0:self.half_in_size])
        x2 = self.fc_spinal_layer2(torch.cat([ x[:,self.half_in_size:2*self.half_in_size], x1], dim=1))
        x3 = self.fc_spinal_layer3(torch.cat([ x[:,0:self.half_in_size], x2], dim=1))
        x4 = self.fc_spinal_layer4(torch.cat([ x[:,self.half_in_size:2*self.half_in_size], x3], dim=1))


        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)


        x = self.fc_out(x)
        return x

class Vgg16(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.vnet = vgg16(pretrained=True)
        self.vnet.classifier = nn.Sequential(
        nn.Linear(25088, output_size))
    def forward(self, x):
        return self.vnet(x)