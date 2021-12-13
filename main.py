import torch
import torch.nn as nn
import numpy as np
import os,sys
import argparse
import gc
import torchvision
from tqdm import tqdm
from Config import CnnConfig, RnnConfig, TrainConfig
from preprocess import one_hot
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

from model import cnn, MobileNetv2,SpinalNet_ResNet, Vgg16

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
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output= model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()  # sum up batch loss
    test_loss /= len(test_loader.dataset)
    print('TEST loss is ', test_loss)
    return test_loss



def main( LR: float, EPOCH: int, OPTIMIZER, MODEL):
    
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
    print(os.getcwd())
    working_path = os.getcwd()


    train_path = working_path + '/data/train'
    test_path = working_path + '/data/test'
    valid_path = working_path + '/data/valid'
    labels = os.listdir(train_path)

    # can change transform here
    train_transform=transforms.Compose([
            transforms.RandomRotation(10),      # rotate +/- 10 degrees
            transforms.RandomHorizontalFlip(),  # reverse 50% of images
            transforms.Resize(224),             # resize shortest side to 50 pixels
            # transforms.CenterCrop(25),         # crop longest side to 50 pixels at center
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

    #======================================================================================================
    # model = cnn().float().to(device)
    model = MODEL().to(device)

    # we use crossEntrophy loss here, because we are doing multi class classfication
    train_cg = TrainConfig(EPOCH=EPOCH, LR=LR, loss_function=nn.CrossEntropyLoss,
                           optimizer=OPTIMIZER)
    EPOCH = train_cg.EPOCH
    optimizer = train_cg.optimizer(model.parameters(), lr=train_cg.LR)
    loss_func = train_cg.loss_function()

    print('start train, device is ', device)
    train_record = []
    test_record = []
    acc_record = []
    # later deal with this part
    # print('dataset pool:', train_data.pooling, 'batch', train_loader.batch_size)
    # print('reg_lambda ', reg_lambda)
    # print('cnn1, cnn2, rnn', cnn1_cg.__dict__, cnn2_cg.__dict__, rnn_cg.__dict__)

    for epoch in tqdm(range(EPOCH)):
        print('epoch: ', epoch)
        for i, batch in enumerate(train_loader):
            images, labels = batch
            labels = one_hot(labels, num_cls=100)

            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            loss = loss_func(output, labels)
            reg = torch.tensor(0.).to(device)

            # if epoch % (EPOCH // 10) == 0 and step == 0:
            #     plot_to_png(output, b_y, False)
            if i % 10 == 0:
                print('training loss is ', loss.item())
                train_record += [loss.item()]
                y_pred = output.argmax(axis=1).cpu()
                y_label = labels.argmax(axis=1).cpu()
                acc = np.mean(y_pred.numpy() == y_label.numpy())
                acc_record += [acc]
                print('accracy is: ', acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

        if epoch % (EPOCH // 20) == 0:

            test_loss = test(model, device, test_loader)
            test_record += [test_loss]
            model.train()
            torch.cuda.empty_cache()


    # torch.save(model, './model/cnnrnn.pkl')
    # np.save('./log/train_record.npy', np.array(train_record)[1:])
    # np.save('./log/test_record.npy', np.array(test_record)[1:])
    return (acc_record, train_record, test_record), model

#####Implementation ###############
if __name__ == '__main__':
    MODEL_performance = ()
    MODEL_list = [cnn,MobileNetv2, SpinalNet_ResNet, Vgg16]
    Modelname = ['cnn','MobileNetv2', 'SpinalNet_ResNet', 'Vgg16']
    optimizer_name = ['Adam','SGD']
    LR = [0.0001, 0.001, 0.01, 0.1]
    optimizer = [torch.optim.Adam, torch.optim.SGD]

    for ind in range(len(MODEL_list)):
        mod = MODEL_list[ind]

        if not os.path.exists('./log/LR/'):
            os.mkdir('./log/LR/')

        if not os.path.exists(os.path.join('./log/LR/', Modelname[ind])):
            os.mkdir(os.path.join('./log/LR/', Modelname[ind]))
        
        for  i in LR:
            MODEL_performance, model = main(MODEL=mod, LR= i, EPOCH=100, OPTIMIZER=torch.optim.Adam)
            np.save(os.path.join('./log/LR/' ,Modelname[ind], 'acc_record_'+ str(i) +'.npy'), np.array(MODEL_performance[0])[1:])
            np.save(os.path.join('./log/LR/' ,Modelname[ind], 'train_record_'+ str(i) +'.npy'), np.array(MODEL_performance[1])[1:])
            np.save(os.path.join('./log/LR/' ,Modelname[ind], 'test_record_'+ str(i) +'.npy'), np.array(MODEL_performance[2])[1:])
            torch.save(model, os.path.join('./log/LR/', Modelname[ind], 'model_' + str(i) + '.pkl'))

        if not os.path.exists('./log/OPT/'):
            os.mkdir('./log/OPT/')

        if not os.path.exists(os.path.join('./log/OPT/', Modelname[ind])):
            os.mkdir(os.path.join('./log/OPT/', Modelname[ind]))
        
        for  j in range(len(optimizer_name)):
            MODEL_performance, model = main(MODEL=mod, LR= 0.001, EPOCH=100, OPTIMIZER=optimizer[j])
            np.save(os.path.join('./log/OPT/' ,Modelname[ind], 'acc_record_'+optimizer_name[j]+'.npy'), np.array(MODEL_performance[0])[1:])
            np.save(os.path.join('./log/OPT/' ,Modelname[ind], 'train_record_'+optimizer_name[j]+'.npy'), np.array(MODEL_performance[1])[1:])
            np.save(os.path.join('./log/OPT/' ,Modelname[ind], 'test_record_'+optimizer_name[j]+'.npy'), np.array(MODEL_performance[2])[1:])
            torch.save(model, os.path.join('./log/OPT/', Modelname[ind], 'model_' + str(j) + '.pkl'))

        

         
    





    

