import torch
import torch.nn as nn
import numpy as np
import os,sys
import argparse
import gc
from tqdm import tqdm

from Config import CnnConfig, RnnConfig, TrainConfig
# from Dataset import Dataset
# from preprocess import split_dataset
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from model import cnn
import matplotlib.pyplot as plt


def plot_to_png(output, target, isTest):
    mod = 'test' if isTest else 'train'
    fig, ax = plt.subplots(1)
    plot(ax, output.detach().to('cpu').numpy()[0].reshape(-1),
         target.detach().to('cpu').numpy()[0].reshape(-1),
         None)
    plt.savefig('./log/plots/' + str(epoch) + '_' + mod + '.png')
    plt.close()


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target, pre_target in test_loader:
            data, target = data.to(device), target.to(device)
            pre_target = pre_target.to(device)
            # output, _ = model(data, pre_target)
            output, _ = model(pre_target)
            test_loss += nn.CrossEntropyLoss()(output, target).item()  # sum up batch loss
            plot_to_png(output, target, True)
    test_loss /= len(test_loader.dataset)
    print('TEST loss is ', test_loss)
    return test_loss


if __name__ == '__main__':
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    parser = argparse.ArgumentParser()
    # parser.add_argument('--float', type=bool, default=True)
    parser.add_argument('--log_file', type=bool, default=True)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # args.log_file = True
    # if args.log_file:
    #     sys.stdout = open('./log/out.txt', 'w')
    #     sys.stderr = open('./log/err.txt', 'w')
    print("yes")
    # no need to split, we already have train and test set in different folders

    # load dataset
    working_path = 'D:/Projects/bird classification'
    train_path = working_path + '/train'
    test_path = working_path + '/test'
    valid_path = working_path + '/valid'
    labels = os.listdir(train_path)
    
    # can change transform here
    train_transform=transforms.Compose([
            transforms.RandomRotation(10),      # rotate +/- 10 degrees
            transforms.RandomHorizontalFlip(),  # reverse 50% of images
            transforms.Resize(25),             # resize shortest side to 50 pixels
            transforms.CenterCrop(25),         # crop longest side to 50 pixels at center
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
    ])

    trainset = ImageFolder(train_path, transform = train_transform)
    testset = ImageFolder(test_path, transform = train_transform)
    validset = ImageFolder(valid_path, transform = train_transform)

    # load dataset into batches
    batch_size = 64
    train_loader = DataLoader(trainset, batch_size, shuffle = True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(validset, batch_size, num_workers=2, pin_memory=True)
    test_loader = DataLoader(testset, batch_size, num_workers=2, pin_memory=True)

    # we can modify this part later, after finishing training the baseline model
    # rnn_cg = RnnConfig(embed_in=5, embed_out=5, hidden_size=8, num_layers=1, batch_first=True)
    # cnn1_cg = CnnConfig(in_channels=1, out_channels=8, kernel_size=(10, 10, 10),
    #                     stride=2, padding=0, pooling=(2, 8, 8))
    # cnn2_cg = CnnConfig(in_channels=8, out_channels=16, kernel_size=(10, 10, 10),
    #                     stride=2, padding=0, pooling=(2, 8, 8))

    # model = CnnRnn(cnn1_cg, cnn2_cg, rnn_cg).float().to(device)
    # model = CnnRnn(rnn_cg).float().to(device)

    model = cnn().float().to(device)
    
    # we use crossEntrophy loss here, because we are doing multi class classfication
    train_cg = TrainConfig(EPOCH=101, LR=0.01, loss_function=nn.CrossEntropyLoss,
                           optimizer=torch.optim.Adam)
    EPOCH = train_cg.EPOCH
    optimizer = train_cg.optimizer(model.parameters(), lr=train_cg.LR)
    loss_func = train_cg.loss_function()
    reg_lambda = 0.0

    print('start train, device is ', device)
    train_record = []
    test_record = []

    # later deal with this part
    # print('dataset pool:', train_data.pooling, 'batch', train_loader.batch_size)
    # print('reg_lambda ', reg_lambda)
    # print('cnn1, cnn2, rnn', cnn1_cg.__dict__, cnn2_cg.__dict__, rnn_cg.__dict__)

    for epoch in range(EPOCH):
        print('epoch: ', epoch)
        batch_count = 0
        for batch in tqdm(train_loader):
            images, labels = batch
            logits = model(images)

            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            loss = loss_func(output, labels)
            reg = torch.tensor(0.).to(device)

            # if epoch % (EPOCH // 10) == 0 and step == 0:
            #     plot_to_png(output, b_y, False)
            batch_count+=1
            if batch_count % 50 == 0:
                print('training loss is ', loss.item())
                train_record += [loss.item()]

            y_pred = torch.max(logits,1)[1]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

        # if epoch % (EPOCH // 20) == 0:
        #     test_loss = test(model, device, test_loader)
        #     test_record += [test_loss]
        #     model.train()
        #     torch.cuda.empty_cache()

    torch.save(model, './model/cnnrnn.pkl')
    np.save('./log/train_record.npy', np.array(train_record)[1:])
    np.save('./log/test_record.npy', np.array(test_record)[1:])
