import math
import os
import random
import shutil
import re
import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 2300000000
static_cols = -1
static_rows = -1


def del_files(dir_path):

    for root, dirs, files in os.walk(dir_path, topdown=False):

        for name in files:
            os.remove(os.path.join(root, name))

        for name in dirs:
            os.rmdir(os.path.join(root, name))
    print("clear done!!")



def qg(orgPath):

    print(orgPath)
    global static_cols
    global static_rows
    DES_HEIGHT = 64#1500
    DES_WIDTH = 64#1500

    path_img = orgPath
    src = cv2.imread(path_img)
    (width, height, depth) = src.shape
    '''height = src.shape[0]
    width = src.shape[1]'''
    padding_img = np.random.randint(0, 255, size=(height, width, 3)).astype(np.uint8)
    padding_img[0:height + 0, 0:width + 0] = src
    img = padding_img
    pic = np.zeros((DES_WIDTH, DES_HEIGHT, depth))
    num_width = int(width / DES_WIDTH)
    num_length = int(height /  DES_HEIGHT)
    static_rows=num_length
    static_cols=num_width
    cols = DES_WIDTH
    rows = DES_HEIGHT
    save_path = "E:/data/caijian/".format(cols, rows)
    filename = os.path.split(path_img)[1]
    for i in range(0, num_width):
        for j in range(0, num_length):
            name_ID = random.randint(1, 10000000)
            pic = src[i * DES_WIDTH: (i + 1) * DES_WIDTH, j * DES_HEIGHT: (j + 1) * DES_HEIGHT, :]
            dd=img[j * rows:(j + 1) * rows, i * cols:(i + 1) * cols, :]
            dst = cv2.resize(dd, (512, 512))
            cv2.imwrite(
                save_path + os.path.splitext(filename)[0] + '_' + str(j) + '_' + str(i) + os.path.splitext(filename)[1],
                dst)
            #cv2.imwrite(pic_target + result_path, dst)



def merge_picture(merge_path, num_of_cols, num_of_rows, target_path, file_name):
    filename = os.listdir(merge_path)
    if filename[0] == '.DS_Store':
        filename.remove('.DS_Store')
    full_path = os.path.join(merge_path, filename[0])
    shape = cv2.imread(full_path).shape
    cols = shape[1]
    rows = shape[0]
    channels = shape[2]

    dst = np.zeros((rows * num_of_rows, cols * num_of_cols, channels), np.uint8)
    for i in range(len(filename)):
        full_path = os.path.join(merge_path, filename[i])
        img = cv2.imread(full_path, -1)
        cols_th = int(full_path.split("_")[-1].split('.')[0])
        rows_th = int(full_path.split("_")[-2])
        roi = img[0:rows, 0:cols, :]
        dst[rows_th * rows:(rows_th + 1) * rows, cols_th * cols:(cols_th + 1) * cols, :] = roi
        dd = cv2.resize(dst, (512, 512))
    cv2.imwrite(target_path + "merge-" + file_name, dd)



def cp(real_org_file_path, real_org_file_name, org_path, fin_path, org_file_name, fin_file_name):
    from PIL import Image
    img = Image.open(org_path + org_file_name)
    org_img = Image.open(real_org_file_path + real_org_file_name)
    region = img.crop((0, 0, org_img.width, org_img.height))
    region.save(fin_path + fin_file_name)


def round_read_file(file_path):
    image_name_list = []
    for file_name in os.listdir(file_path):
        if file_name != '.DS_Store':
            print( str(file_path) + str(file_name))
            image_name_list.append(str(file_name))
    print(str(len(image_name_list)))
    return image_name_list



def file_corp_file(name,old_file_path,new_folder_path):
    del_files("/disk/sdc/")
    print(name)
    image = os.listdir(old_file_path)
    img_tests_path = []
    name = main_file_name.split('.')[0]
    for i in range(len(image)):
        img_tests_path.append(old_file_path + '/' + image[i])

        if re.match(name,image[i]):
            print(image[i])
            picture = cv2.imread(img_tests_path[i])
            result = new_folder_path + '/' + image[ i]
            print(result)
            cv2.imwrite(result, picture)



if __name__ == '__main__':
    main_tmp_path = '/disk/sdc/label/label3.5/'
    new_path='/disk/sdc/camdata/yy/'
    main_org_path = '/disk/sdc/data/image/image3.5/'
    main_merge_path = '/disk/sdc/camdata/result/'
    main_fin_path = '/disk/sdc/Lkongdong/camdata/er/'
    main_img_name_list = round_read_file(main_org_path)
    i = 1
    k = 0
    for main_file_name in main_img_name_list:
        file_corp_file(main_file_name, main_tmp_path, new_path)
        static_rows = 4
        static_cols = 4
        merge_picture( new_path, static_rows,static_cols, main_merge_path, main_file_name)
        print( str(i) + "/" + str(len(main_img_name_list)))
        cp(main_org_path, main_file_name, main_merge_path, main_fin_path, str("merge-" + main_file_name),
           main_file_name)
        print(str(i) + "/" + str(len(main_img_name_list)))
        static_rows = -1
        static_cols = -1
        i = i + 1
        k = k +4
    print("done!!!!")

