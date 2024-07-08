import glob
import numpy as np
import torch.nn as nn
import torch
import os
import cv2
from model.unet_model import SoftDiceLoss, UNet
# from model.trans_modelV1 import SoftDiceLoss, UNet
# from util.vision import feature_vis, feature_64

# from model.unet_model import RSTN
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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

def pred(epoch):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(n_channels=1, n_classes=1)
    net.to(device=device)
    # net.load_state_dict(torch.load('/disk/sdb/lijiaming/Model/unet/checkpoint2/cp_' + checkpoint + '/epoch_' + str(epoch) + '_best_model.pth', map_location=device))
    net.load_state_dict(torch.load('/disk/sdc/unet/checkpoint/epoch_' + str(epoch) + '_best_model.pth', map_location=device))
    net.train()
    file_list = os.listdir("/disk/sdc/unet/data/test/image/")
    label_fille_list = os.listdir("/disk/sdc/unet/data/test/label/")
    tests_path = glob.glob('/disk/sdc/unet/data/test/image/*.jpg')
    label_tests_path = glob.glob('/disk/sdc/unet/data/test/label/*.jpg')
    file_list.sort()
    label_fille_list.sort()
    print(len(label_fille_list), len(file_list))
    for i in range(len(file_list)):
        path = "/disk/sdc/unet/data/test/image/" + file_list[i]
        label_path = "/disk/sdc/unet/data/test/label/" + label_fille_list[i]
        tests_path[i] = path
        label_tests_path[i] = label_path

    dice = 0
    # for test_path in tests_path:
    with torch.no_grad():
        for i in range(len(tests_path)):
            test_path = tests_path[i]
            label_test_path = label_tests_path[i]
            save_res_path =  "/disk/sdc/unet/data/test/label1/"+test_path.split('/')[-1]
            # save_res_path2 = test_path.split('.')[0] + '_coarse.jpg'
            img = cv2.imread(test_path)
            label = cv2.imread(label_test_path)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            label = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY)
            img = img.reshape(1, 1, img.shape[0], img.shape[1])
            label = label.reshape(1, 1, label.shape[0], label.shape[1])
            if label.max() > 1:
                label = label / 255
            img_tensor = torch.from_numpy(img)
            label_tensor = torch.from_numpy(label)
            img_tensor = img_tensor.to(device=device, dtype=torch.float32)
            label_tensor = label_tensor.to(device=device, dtype=torch.float32)
            # predict
            pred = net(img_tensor)
            # print()
            # print("pred.shape", pred.shape)
            #DSC
            dice += diceCoeff(pred, label_tensor)
            print("dice =", diceCoeff(pred, label_tensor))
            pred = np.array(pred.data.cpu()[0])[0]
            pred[pred >= 0.5] = 255
            pred[pred < 0.5] = 0
            save_res_path = save_res_path.replace("val", "CD_pl_test")
            cv2.imwrite(save_res_path, pred)
            # cv2.imwrite(save_res_path + "_sigmoid", h)
            # feature_64(feature_map)
            # feature_vis(feature_map)
        print("all_dice = ", dice.item())
        print("avg_dice = ", dice.item() / 1100)
        with open('/disk/sdc/unet/bestepoch/CD.txt', 'a') as f:
            f.write(" epoch " + str(epoch) + " avg_dice = " + str(dice.item() / 1100) + '\n')

if __name__ == "__main__":
    # list = [10, 12, 14, 16, 20, 30, 40]
    list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 195, 196, 197, 198, 199]
    for i in list:
        pred(i)