import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random
import numpy as np

class ISBI_Loader(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        file_list = os.listdir(self.data_path + "image")
        self.imgs_path = glob.glob(os.path.join(data_path,  'image/*.jpg'))
        for i in range(len(file_list)) :
            path = self.data_path + "image/" + file_list[i]
            self.imgs_path[i] = path
        # print(self.imgs_path)
        # print(self.imgs_path[0])

    def augment(self, image, flipCode):

        if flipCode == -1:
            back = image[:, ::-1, :].copy()
        elif flipCode == 0:
            back = image[:, :, ::-1].copy()
        elif flipCode == 1:
            back = image[:, ::-1, ::-1].copy()
        else:
            back = image
        return back

    def __getitem__(self, index):

        image_path = self.imgs_path[index]
        label_path = image_path.replace('image', 'label')
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        label[label < 200] = 0
        label[label >= 200] = 1
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
        return image, label, image_path

    def __len__(self):
        return len(self.imgs_path)


if __name__ == "__main__":
    isbi_dataset = ISBI_Loader(r"../data/train/")
    print("number：", len(isbi_dataset))
    print("**************")
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=2,
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)
