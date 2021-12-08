import torch
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import make_grid
import os
from tqdm.auto import tqdm
from model import MobileNetv2
import matplotlib.pyplot as plt
from Config import TrainConfig
import pandas as pd
import numpy as np
import time
import copy
from torch.optim import lr_scheduler

##########################################################################################################################################################
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    dataset_sizes= datasizes
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    test_token=0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid','test']:
        #validation here    
            
            '''
            Test when a better validation result is found
            '''

            if test_token == 0 and phase == 'test':
                continue
            test_token =0
            
            
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                test_token =1


        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

##########################################################################################################################################################
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    crop_size = 224

    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(crop_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    test_transform = transforms.Compose([
        transforms.Resize(230),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    working_path = os.getcwd()
    train_path = working_path + '/data/train'
    test_path = working_path + '/data/test'
    valid_path = working_path + '/data/valid'

    train_set = datasets.ImageFolder(train_path, transform=train_transform)
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    val_set = datasets.ImageFolder(valid_path, transform=test_transform)
    valloader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    test_set = datasets.ImageFolder(test_path, transform=test_transform)
    testloader = DataLoader(test_set, shuffle=True)

    dataloaders = {
        "train": trainloader,
        "test": testloader,
        "valid": valloader
    }

    datasizes = {
        "train": len(train_set),
        "test": len(test_set),
        "valid":len(val_set)
    }
    CLASSES = list(train_set.class_to_idx.keys())

    label2cat = train_set.classes
    feature, target = next(iter(trainloader))

    model = MobileNetv2(output_size=crop_size).to(device)
    criterion = nn.NLLLoss()
    optimizer_def = optim.AdamW(model.parameters(), lr=0.001)
    exp_lr_sc = lr_scheduler.StepLR(optimizer_def, step_size=7, gamma=0.1)

    model_ft = train_model(model, criterion, optimizer_def, exp_lr_sc, num_epochs=5)









