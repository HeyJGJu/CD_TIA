
import os

import cv2
import operator

def img_input(img_path, label_path):
    img = cv2.imread(img_path)
    label = cv2.imread(label_path)
    return (img, label)


def img_size(img):
    white = 0
    black = 0
    list1 = [255, 255, 255]
    list2 = [0, 0, 0]
    for x in img:
        for y in x:
            if operator.eq(y.tolist(), list1) == True:
                white = white + 1
            elif operator.eq(y.tolist(), list2) == True:
                black = black + 1
    return (white, black)


def size_same(img, label):
    size = 0
    list = [255, 255, 255]
    for x1, x2 in zip(img, label):
        for y1, y2 in zip(x1, x2):
            if operator.eq(y1.tolist(), y2.tolist()) & operator.eq(y1.tolist(), list):
                size = size + 1
    return size


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')
    img_path = r"/disk/sdc/unet/data/test/label"
    label_path = r"/disk/sdc/unet/data/test/label1"
    img, label = img_input(img_path, label_path)

    Dice = 0
    num = 0
    max = 0
    min=1
    for image in os.listdir(img_path):
        img = os.path.join(img_path, image)
        label = os.path.join(label_path, image)

        img, label = img_input(img, label)


        white1, black1 = img_size(img)
        white2, black2 = img_size(label)
        size = size_same(img, label)
        if((white1 + white2) != 0.0):
          dice = 2 * size / (white1 + white2)  
        else:  
          dice =0     
        # print("white1:", white1, "black1:", black1)
        # print("white2:", white2, "black2:", black2)
        # print("same size:", size)
        print(image + " dice:", dice)
        #Dice = Dice + dice
        #num = num + 1
        if(dice>= 0.0):
            Dice = Dice + dice
            num = num + 1
        if (dice != 0.0):
            if (dice <min):
                min= dice

        if (dice > max):
            max = dice
    average = Dice / num
 
    print("average Dice:", average)
    print("max dice: ", max)
    print("min dice: ", min)
    print("num: ", num)
