
import os
import random
import shutil
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

def crop(pred, img, label=None):
    left = right = top = bottom = 20

    (N, C, W, H) = pred.shape
    minA = 0
    maxA = W
    minB = 0
    maxB = H
    binary_mask = pred
    mask = torch.zeros(size=(N, C, W, H))
    cur_mask = binary_mask[0, 0, :, :]
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    arr = torch.nonzero(cur_mask)
    if (arr.shape[0] != 0):
        minA = arr[:, 0].min().item()
        maxA = arr[:, 0].max().item()
        minB = arr[:, 1].min().item()
        maxB = arr[:, 1].max().item()
    bbox = [int(max(minA, 0)), int(min(maxA, W)), \
            int(max(minB, 0)), int(min(maxB, H))]
    mask[0, 0, bbox[0]: bbox[1], bbox[2]:bbox[3]] = 1
    #img = img * mask
    crop_img = img[:, :, bbox[0]:bbox[1], bbox[2]:bbox[3]]
    #print(bbox)
    crop_img=tensor_to_image(crop_img)
    dct=cv2.resize(crop_img,(512,512))
   
    return crop_img, bbox
def tensor_to_image(tensor):
    if tensor.dim()==4:
        tensor=tensor.squeeze(0)
    tensor=tensor.permute(1,2,0)
    tensor=tensor.mul(255).clamp(0,255)
    tensor=tensor.cpu().numpy().astype('uint8')  ###
    return tensor


def uncrop(crop_info, cropped_image):

    bbox = crop_info
    #image[0, 0, bbox[0]: bbox[1], bbox[2]: bbox[3]] = cropped_image
    return cropped_image

if __name__ == "__main__":
    #pred_path = "D:/data/label"
    pred_path = "/disk/sdc/AMatrix/data/ill/label"
    #img_path = "D:/data/image"
    img_path = "/disk/sdc/AMatrix/data/ill/addWeight"
    #first_path = "D:/data/firstpicture"
    result_path = "/disk/sdc/AMatrix/data/ill/cut_addweight"

    pred = os.listdir(pred_path)  # 1.jpg
    image = os.listdir(img_path)
    #first = os.listdir(first_path)
    label_tests_path = []
    img_tests_path = []
    first_tests_path = []

    #length = len(first)
    name_slice = 0

    for i in range(len(pred)):
        label_tests_path.append(pred_path + '/' + pred[i])
    for i in range(len(image)):
        img_tests_path.append(img_path + '/' + image[i])
    '''for i in range(length):
        first_tests_path.append(first_path + '/' + first[i])'''
    for i in range(len(label_tests_path)):
        slice_num = 1
        fist_img_list = []
        picture = cv2.imread(img_tests_path[i], flags=0)
        label = cv2.imread(label_tests_path[i], flags=0)
        '''for j in range(slice_num):
            k = random.randint(0, length -  1)
            slice = cv2.imread(first_tests_path[k], flags=0)
            slice = slice.reshape(1, 1, slice.shape[0], slice.shape[1])
            slice_tensor = torch.from_numpy(slice)
            fist_img_list.append(slice_tensor)'''

        picture = picture.reshape(1, 1, picture.shape[0], picture.shape[1])
        label = label.reshape(1, 1, label.shape[0], label.shape[1])
        if label.max() > 1:
            label = label / 255
        #tensor
        picture_tensor = torch.from_numpy(picture)
        label_tensor = torch.from_numpy(label)

        crop_img, crop_info = crop(label_tensor, picture_tensor)
        img_name=result_path+'/'+pred[i]
        cv2.imwrite(img_name,crop_img)
        '''for j in range(slice_num):
            uncrop_img_save_path = img_tests_path[i].replace('image', 'uncropImage')
            uncrop_img_save_path = uncrop_img_save_path.split('.')[0] + '_index' + str(name_slice) + '.jpg'
            uncrop_lab_save_path = uncrop_img_save_path.split('.')[0] + '_index' + str(name_slice) + '.jpg'
            uncrop_lab_save_path = uncrop_img_save_path.replace('uncropImage', 'uncropLabel')
            #uncrop_img = uncrop(crop_info, crop_img, fist_img_list[j])
            uncrop_img = uncrop(crop_info, crop_img)
            uncrop_img = np.array(uncrop_img.data.cpu()[0])[0]
            cv2.imwrite(uncrop_img_save_path, uncrop_img)
            shutil.copyfile(label_tests_path[i], uncrop_lab_save_path)
            name_slice = name_slice + 1'''



