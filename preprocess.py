import torch
import cv2
import numpy as np
from sklearn.decomposition import PCA
import os

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms


def one_hot(labels, num_cls):
    oneHots = torch.zeros((len(labels), num_cls))
    oneHots[range(len(oneHots)), labels] = 1
    return oneHots

def pca_image(imgs: np.array, k):
    h, w, c = imgs.shape
    imgs = imgs.reshape(h, -1)
    pca = PCA(k)
    tmp = pca.fit_transform(imgs)
    tmp = pca.inverse_transform(tmp)
    res = np.array(tmp.reshape(h, w, c))
    return res, pca.explained_variance_ratio_

def save_pac_image():
    for folder in os.listdir('./data'):
        if not os.path.isdir(os.path.join('./data', folder)) or folder.endswith('_FE'):
            continue
        pca_folder = folder + '_FE'
        if not os.path.exists(os.path.join('./data', pca_folder)):
            os.mkdir(os.path.join('./data', pca_folder))
        print('-----', folder)
        for cls in os.listdir(os.path.join('./data', folder)):
            if not os.path.isdir(os.path.join('./data', folder, cls)):
                continue
            if not os.path.exists(os.path.join('./data', pca_folder, cls)):
                os.mkdir(os.path.join('./data', pca_folder, cls))
            print('======', os.path.join('./data', folder, cls))
            for file in os.listdir(os.path.join('./data', folder, cls)):
                if file.endswith('.jpg'):
                    ori_img = cv2.imread(os.path.join('./data', folder, cls, file))
                    k = find_opt_pca(ori_img, range(200, 224))
                    pca_img, _ = pca_image(ori_img, k)
                    cv2.imwrite(os.path.join('./data', pca_folder, cls, file), pca_img)
                    print(file, 'n_comp=', k)


def find_opt_pca(imgs, n_comp):
    res = []
    for i in n_comp:
        _, score = pca_image(imgs, i)
        res += [sum(score)]
        # print('n_componet: ', i, ' explained ratio: ', sum(score))
    dif = np.diff(np.diff(res))
    return n_comp[np.argmax(dif)]

if __name__ == '__main__':
    # img = cv2.imread('1.jpg')
    save_pac_image()
    # transform = transforms.Compose([
    #     transforms.ToTensor()
    # ])
    # imgs = ImageFolder('./data/test/', transform=transform)
    # print(imgs[0][0], imgs[0][1])
    # train_loader = DataLoader(imgs, 2, shuffle = False, num_workers=1)
    # print(DataLoader)
    # for i, batch in enumerate(train_loader):
    #     images, labels = batch
    #     labels = one_hot(labels, num_cls=100)
    #     print(images[0].numpy().shape)
    #     cv2.imshow('ori', np.moveaxis(images[0].numpy(), [0, 1, 2], [2, 0, 1]))
    #     cv2.waitKey(0)
    #
    #     pca_img, _ = pca_image(np.moveaxis(images[0].numpy(), [0, 1, 2], [2, 0, 1]), 200)
    #     print(pca_img.shape)
    #     cv2.imshow('im', pca_img)
    #     cv2.waitKey(0)
    #     print(images.shape, labels)
    #
    # imgs = np.array([img for i in range(150)])
    # imgs = np.random.randn(250, 224, 224, 3)
    # cv2.imshow('m', img)
    # cv2.waitKey(0)

    # res, _ = pca_image(imgs, 100)
    # print('---------', res.shape)
    # cv2.imshow('m1',res[0])
    # cv2.waitKey(0)

    # k = find_opt_pca(imgs, 224, 20)
    # print(k)




