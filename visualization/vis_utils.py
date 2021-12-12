import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import cv2
from model import CnnRnn



def plot(ax, label: np.array, predict: np.array, imgs: np.array):
    '''
    plot label predict and movie align to time using
    :param label:
    :param predict:
    :param imgs:
    :return:
    '''

    ax.plot(label, color='orange', linewidth=0.2)
    ax.plot(predict, color='blue', linewidth=0.2)
    ax.set_xlabel('responds')
    ax.set_xlabel('time')

def plot_param(model):
    fig, ax = plt.subplots(2, 2)
    for p in model.parameters():
        if p.requires_grad:
            ax[0, 0].title.set_text(p.name)
            print(p.data.shape, type(p.data), p.name)
            data = p.data.reshape(8, 15, 15)
            ax[0, 0].imshow(data[0])
            ax[0, 1].imshow(data[1])
            ax[1, 0].imshow(data[2])
            ax[1, 1].imshow(data[3])
            plt.show()
def plot_input(responds_path):
    fig, ax = plt.subplots(1)
    responds = np.load(responds_path)[:600]
    ax.plot(responds)
    plt.show()


if __name__ == '__main__':
    # fig, ax = plt.subplots(1)
    # x = np.arange(10)
    # y = np.random.rand(10, 1)
    # plot(ax, x, y, None)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('./model/cnnrnn.pkl', map_location=torch.device(device))
    model.eval()
    # plot_param(model)

    # show_imgs("./Data/super_fruit_movie.npy")
    # plt.show()
