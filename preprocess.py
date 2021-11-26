
import cv2
import numpy as np
from sklearn.decomposition import PCA

from torchvision.datasets import ImageFolder


def get_dict():
    '''
    from 'class_dict.csv', return a dictory that key is the name of the bird class and the value is the index number.
    '''
    pass


def read_data_image(folder: str):
    '''
    read image data in folder as numpy array. Return the numpy array 
    Args:
        folder: name of the folder
    Return:
        numpy array of size (N x H x W x 3). where N is the number of images, H is the height, and W is the width, 3 is the channels
    
    '''


def save_data_label(folder: str):
    '''
    get the one-hot label of the folder as numpy array. Return the numpy array 
    Args:
        folder: name of the folder
    Return:
        numpy array of size (N x C). where N is the number of images, C is the number of class
    
    '''

def pca_image(imgs: np.array, k):
    N, h, w, c = imgs.shape
    imgs = imgs.reshape(N, -1)
    pca = PCA(k)
    tmp = pca.fit_transform(imgs)
    tmp = pca.inverse_transform(tmp)
    res = np.array(tmp.reshape(N, h, w, c), dtype=np.uint8)
    return res, pca.explained_variance_ratio_

def find_opt_pca(imgs, max_n_comp, step):
    res = []
    for i in range(1, max_n_comp, step):
        _, score = pca_image(imgs, i)
        res += [sum(score)]
        print('n_componet: ', i, ' explained ratio: ', sum(score))
    dif = np.diff(np.diff(res))
    return np.argmax(dif) * step

if __name__ == '__main__':
    img = cv2.imread('1.jpg')
    # imgs = ImageFolder('./data')
    # for img, label in imgs:
    #     print(type(img), label)

    # imgs = np.array([img for i in range(150)])
    imgs = np.random.randn(250, 224, 224, 3)
    # cv2.imshow('m', img)
    # cv2.waitKey(0)

    # res, _ = pca_image(imgs, 100)
    # print('---------', res.shape)
    # cv2.imshow('m1',res[0])
    # cv2.waitKey(0)

    k = find_opt_pca(imgs, 224, 20)
    print(k)




