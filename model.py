import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

class cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(# 
            nn.Conv2d(3,10,kernel_size=3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(10,15,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # 

            nn.Conv2d(15, 20, kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(20,20,kernel_size=3,stride=1,padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2,2), 

            nn.Conv2d(20,25,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(25,25,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2), 

            nn.Flatten(),
            nn.Linear(225,128),
            nn.Dropout(0.25),
            nn.Linear(128,64),
            nn.Dropout(0.25),
            nn.Linear(64,100)
        )

    def forward(self,x):
        return self.feature(x)


class MobileNetv2(nn.Module):
  def __init__(self, output_size):
    super().__init__()
    self.mnet = mobilenet_v2(pretrained=True)
    self.freeze()

    self.mnet.classifier = nn.Sequential(
        nn.Linear(1280, output_size),
        nn.LogSoftmax(1)
    )

  def forward(self, x):
    return self.mnet(x)
  



