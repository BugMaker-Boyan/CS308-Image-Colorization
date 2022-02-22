import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch


class ImageNetDataset(Dataset):
    def __init__(self, is_train):
        if is_train:
            root_path = "tiny-imagenet-200/train"
        else:
            root_path = "tiny-imagenet-200/test"

        dirs = os.listdir(root_path)
        self.images = []
        for d in dirs:
            path = root_path + "/" + d + "/images"
            self.images.extend(path + '/' + i for i in os.listdir(path))

    def __getitem__(self, item):
        path = self.images[item]
        image = cv2.imread(path)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        L, a, b = cv2.split(lab)
        ori = np.stack((L / 128.0 - 1, a / 128.0 - 1, b / 128.0 - 1), axis=0)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = np.reshape(gray, (-1, 64, 64)) / 128.0 - 1
        return torch.tensor(gray, dtype=torch.float), torch.tensor(ori, dtype=torch.float)

    def __len__(self):
        return len(self.images)
