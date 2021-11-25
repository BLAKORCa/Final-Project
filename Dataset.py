import torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import TypeVar

from preprocess import pool2d
T = TypeVar('T', np.array, torch.tensor)



class RetinaDataset(Dataset):
    def __init__(self, imgs_path: str, responds_path: str, is_float: bool) -> None:
        imgs = np.load(imgs_path)
        responds = np.load(responds_path)
        assert imgs.shape[0] == responds.shape[0]

        imgs = torch.tensor(imgs)
        responds = torch.tensor(responds)

        if is_float:
            imgs = imgs.float()
            responds = responds.float()
        self.imgs = imgs
        self.responds = responds
        self.pooling = -1

    def __getitem__(self, index) -> (T, T):
        return self.imgs[index], self.responds[index], self.pre_responds[index]

    def __len__(self) -> int:
        return self.imgs.shape[0]


if __name__ == '__main__':
    print()