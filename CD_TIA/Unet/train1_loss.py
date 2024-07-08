import cv2
import numpy as np
from model.unet_model import UNet
from model.unet_model import SoftDiceLoss
from util.dataset import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def diceCoeff(pred, gt, image_path, smooth=1e-5, activation='sigmoid'):
    image_name = image_path.split('/')[-1].split('.')[0]
    txt_base_path = r'/disk/sdc/unet/tex/'   #Weight data
    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d")

    pred = activation_fn(pred)
    # print(pred)
    txt_path =  os.path.join(txt_base_path, image_name + '.txt')
    # print('txt_path', txt_path)
    tag = torch.tensor(0.1 * np.loadtxt(txt_path))
    tag = tag.to(device)


    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)
    tag = tag.view(N, -1)
    # print('111111111111', pred_flat.type())
    intersection = (pred_flat * gt_flat * tag).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss = (2 * intersection + smooth) / (unionset + smooth)

    return loss.sum() / N


def train_net(net, device, data_path, epochs=50, batch_size=1, lr=0.00001):
    isbi_dataset = ISBI_Loader(data_path)
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # criterion = nn.BCEWithLogitsLoss()
    # criterion01 = nn.BCEWithLogitsLoss()
    criterion = SoftDiceLoss()
    best_loss = float('inf')
    for epoch in range(epochs):
        net.train()
        for image, label, image_path in train_loader:
            # print(image_path)
            image_path = str(image_path)
            # print(image_path)
            optimizer.zero_grad()
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            pred = net(image)
            if epoch + 1 % 5 == 0:
                pred_tensor = np.array(pred.data.cpu()[0])[0]
                pred_tensor[pred_tensor >= 0.5] = 1
                pred_tensor[pred_tensor < 0.5] = 0
                # save_base_path = r'F:\newlabel\predepoch'
                save_base_path = '/disk/sdc/unet/checkpoint/'
                image_name = image_path.split('/')[-1].split('.')[0]
                save_path = save_base_path + str(epoch)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                image_pred_path = os.path.join(save_path, image_name)
                image_pred_path = image_pred_path + '.jpg'
                # print(image_pred_path)
                # cv2.imwrite(image_pred_path, pred_tensor * 255)
            loss = criterion(pred, label)
            # loss = (1/3) * criterion01(pred, label) + (2/3) * criterion(pred, label)
            dice = diceCoeff(pred, label, image_path)
            print('epoch = ', epoch, 'Loss/train ', loss.item(), 'Dice/train ', dice.item())
            if dice.item() < 0.0001:
                print("=" * 30)
                print("path:", image_path)
                print("=" * 30)
            if loss < best_loss:
                best_loss = loss
                print('--------> best dice:', dice.item())
                torch.save(net.state_dict(), '/disk/sdc/unet/bestepoch/' + str(epoch) + 'best_model.pth')
            loss.backward()
            optimizer.step()
    print("epoch num = :", epoch)


if __name__ == "__main__":
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    net = UNet(n_channels=1, n_classes=1)
    # weights_path = "./pre_model.pth"
    # assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    # net.load_state_dict(torch.load(weights_path))
    net.to(device=device)
    data_path = "/disk/sdc/unet/data/train/"
    train_net(net, device, data_path, epochs=50)
