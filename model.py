import torch
import torch.nn as nn

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
    
class MoblieNet(nn.Module):
    def __init__(self, num_cls):
        super().__init__()
        self.MN = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        self.linear = nn.Linear(1000, num_cls)
        self.drop = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        x = self.MN(x)
        x = self.linear(x)
        x = self.drop(x)
        x = self.softmax(x)
        return x
        