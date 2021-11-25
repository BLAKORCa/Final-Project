import torch
import torch.nn as nn
import numpy as np
import sys
import argparse
import gc

from Config import CnnConfig, RnnConfig, TrainConfig
from Dataset import Dataset
from preprocess import split_dataset
from torch.utils.data import DataLoader
from model.CnnRnn import CnnRnn
from model.Rnn import Rnn
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
            test_loss += nn.MSELoss()(output, target).item()  # sum up batch loss
            plot_to_png(output, target, True)
    test_loss /= len(test_loader.dataset)
    print('TEST loss is ', test_loss)
    return test_loss


if __name__ == '__main__':
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--float', type=bool, default=True)
    parser.add_argument('--log_file', type=bool, default=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.log_file:
        sys.stdout = open('./log/out.txt', 'w')
        sys.stderr = open('./log/err.txt', 'w')

    train_x, train_y, test_x, test_y = split_dataset(
        imgs_path="",
        responds_path="",
        n_splits=5,
        test_fold=0
    )

    print('tx, ty: ', train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    train_x 
    # (N, time_step, channel, height, width)
    train_y 
    # (N, time_step, label)
    test_x 
    test_y 

    # train_x = np.random.rand(1, 1, 1, 500, 500)
    # train_y = np.random.rand(1, 1, 1)
    # test_x = np.random.rand(1, 1, 1, 500, 500)
    # test_x = torch.tensor(test_x)
    # test_y = np.random.rand(1, 1, 1)
    # test_y = torch.tensor(test_y)

    print('train_x: ', train_x.shape, 'train_y: ', train_y.shape)
    train_data = Dataset(imgs=train_x, responds=train_y,
                               is_float=True, pooling=2)
    train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=False, num_workers=1)
    del train_x, train_y
    gc.collect()
    test_data = Dataset(imgs=test_x, responds=test_y,
                              is_float=True, pooling=2)
    test_loader = DataLoader(dataset=test_data, batch_size=16, shuffle=False, num_workers=1)
    del test_x, test_y
    gc.collect()

    rnn_cg = RnnConfig(embed_in=5, embed_out=5, hidden_size=8, num_layers=1, batch_first=True)
    cnn1_cg = CnnConfig(in_channels=1, out_channels=8, kernel_size=(10, 10, 10),
                        stride=2, padding=0, pooling=(2, 8, 8))
    cnn2_cg = CnnConfig(in_channels=8, out_channels=16, kernel_size=(10, 10, 10),
                        stride=2, padding=0, pooling=(2, 8, 8))
    # model = CnnRnn(cnn1_cg, cnn2_cg, rnn_cg).float().to(device)
    model = CnnRnn(rnn_cg).float().to(device)

    train_cg = TrainConfig(EPOCH=101, LR=0.01, loss_function=nn.MSELoss,
                           optimizer=torch.optim.Adam)
    EPOCH = train_cg.EPOCH
    optimizer = train_cg.optimizer(model.parameters(), lr=train_cg.LR)
    loss_func = train_cg.loss_function()
    reg_lambda = 0.0

    print('start train, device is ', device)
    train_record = []
    test_record = []

    print('dataset pool:', train_data.pooling, 'batch', train_loader.batch_size)
    print('reg_lambda ', reg_lambda)
    print('cnn1, cnn2, rnn', cnn1_cg.__dict__, cnn2_cg.__dict__, rnn_cg.__dict__)

    for epoch in range(EPOCH):
        print('epoch: ', epoch)
        for step, (b_x, b_y, b_pre_y) in enumerate(train_loader):
            # print(b_x.shape, 'bx', b_y.shape, 'by')
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            # b_pre_y = b_pre_y.to(device)

            # output, _ = model(b_x, b_pre_y)
            output, _ = model(b_x, None)
            loss = loss_func(output, b_y)
            reg = torch.tensor(0.).to(device)

            if epoch % (EPOCH // 10) == 0 and step == 0:
                plot_to_png(output, b_y, False)
            if step == 0:
                print('training loss is ', loss)
                train_record += [loss.item()]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

        if epoch % (EPOCH // 20) == 0:
            test_loss = test(model, device, test_loader)
            test_record += [test_loss]
            model.train()
            torch.cuda.empty_cache()

    torch.save(model, './model/cnnrnn.pkl')
    np.save('./log/train_record.npy', np.array(train_record)[1:])
    np.save('./log/test_record.npy', np.array(test_record)[1:])
