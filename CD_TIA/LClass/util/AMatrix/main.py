import torch
import matplotlib.pyplot as plt
import torchvision.transforms as tfs
from PIL import Image
import os
import numpy as np
import cv2 as cv
import time

def getAffinity_Matrix(img,name):
    img = img.permute(1,2,0)
    # [width, height]
    affinity = torch.zeros(img.shape[0]*img.shape[1], img.shape[0]*img.shape[1])
    print(affinity.shape)
    img1 = img.reshape(-1, img.shape[-1])
    img_ = torch.sqrt((img1[:,:]**2).sum(dim=-1))
    img2 = img.reshape(-1, img.shape[-1])
    for idx in range(affinity.shape[1]):
        affinity[idx, :] = torch.mul(img1[idx, :], img2[:, :]).sum(dim=-1)
        affinity[idx, :] = affinity[idx, :]/img_[idx]
    for idx in range(affinity.shape[0]):
        #continue
        affinity[:, idx] = affinity[:, idx]/img_[idx]
    print("-----------------")
    print(affinity)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.savefig("/disk/sdc/AMatrix/data/noill/1/"+name,bbox_inches='tight',pad_inches=0)
    return affinity

def display(affinity):
    plt.imshow(affinity)
    #plt.colorbar()
    plt.savefig("/disk/sdc/data/affinity.jpg",bbox_inches='tight',pad_inches=0)
    plt.show()

def process(img_root, rate=16):
    img = Image.open(img_root)
    size = 224//rate
    img = img.resize((size,size))
    plt.imshow(img)
    img = tfs.ToTensor()(img)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = img.to(device)
    return img

img_path="/disk/sdc/AMatrix/data/noill/image"
image = os.listdir(img_path)
img_tests_path = []
for i in range(len(image)):
    img_tests_path.append(img_path + '/' + image[i])
for k in range(len(img_tests_path)):
    img = cv.imread(img_tests_path[k])
    img = process(img_tests_path[k])
    affinity = getAffinity_Matrix(img,str(image[k]))

