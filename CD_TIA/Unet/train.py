from model.unet_model import SoftDiceLoss, UNet
from predict_v2  import pred
# from model.trans_modelV1 import SoftDiceLoss, UNet
# from model.depth_unet_modelV3 import SoftDiceLoss, UNet
# from model.depth_unet_modelV4 import SoftDiceLoss, UNet
# from util.Active_Contour_Loss import Active_Contour_Loss
from util.dataset import ISBI_Loader
# from util.focal_loss import FocalLossV1
from torch import optim
import torch.nn as nn
import torch
import os
import numpy as np
import cv2


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def diceCoeff(pred, gt, smooth=1e-5, activation='sigmoid'):
 
    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d")
 
    pred = activation_fn(pred)
 
    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)
 
    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss = (2 * intersection + smooth) / (unionset + smooth)
 
    return loss.sum() / N

def train_net(net, device, data_path, epochs=50, batch_size=2, lr=0.00001):
    isbi_dataset = ISBI_Loader(data_path)
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-9, momentum=0.9)
    criterion = SoftDiceLoss()
    # best_loss
    best_loss = float('inf')
    for epoch in range(epochs):
        net.train()
        for image, label in train_loader:
            optimizer.zero_grad()
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # cin = cin.to(device=device, dtype=torch.float32)
            # cout = cout.to(device=device, dtype=torch.float32)
            pred = net(image)
            # print(pred.shape, cin.shape)
            # loss = 0.001 * criterion1(label, pred, cin, cout) + 0.999 * criterion(pred, label)
            # loss = 0.625 * criterion(pred, label) + 0.3125 * criterion1(pred, label) + 0.0625 * criterion2(label, pred, cin, cout)
            loss = criterion(pred, label)
            dice = diceCoeff(pred, label)
            print('epoch = ', epoch, 'Loss/train', loss.item(), 'Dice/train', dice.item())
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), '/disk/sdc/unet/best_model.pth')
            loss.backward()
            optimizer.step()
        if (epoch % 10 == 0 or epoch > 190):
            torch.save(net.state_dict(), '/disk/sdc/unet/checkpoint/epoch_' + str(epoch) + '_best_model.pth')


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(n_channels=1, n_classes=1)
    net.to(device=device)
    #data_path = "/home/nwu/dataset/CD_pl/train/"
    data_path = "/disk/sdc/unet/data/train/"
    train_net(net, device, data_path, epochs=200)
