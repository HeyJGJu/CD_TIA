import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

img_path='/disk/sdc/label_eporch46'
image = os.listdir(img_path)
img_tests_path = []
for i in range(len(image)):
    img_tests_path.append(img_path + '/' + image[i])
    frame = cv2.imread(img_tests_path[i])

    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    #lower_blue = np.array([110, 100, 100])
    lower_blue = np.array([100, 90, 90])
    upper_blue = np.array([130, 255, 255])

    lower_green = np.array([50, 100, 100])
    upper_green = np.array([70, 255, 255])

    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    red_mask = cv2.inRange(hsv, lower_red, upper_red)


    blue_res = cv2.bitwise_and(frame, frame, mask=blue_mask)
    green_res = cv2.bitwise_and(frame, frame, mask=green_mask)
    red_res = cv2.bitwise_and(frame, frame, mask=red_mask)


    res = blue_res + green_res + red_res


    frame = frame[:, :, ::-1]
    blue_res = blue_res[:, :, ::-1]
    green_res = green_res[:, :, ::-1]
    red_res = red_res[:, :, ::-1]
    res = res[:, :, ::-1]
    '''plt.figure(figsize=(14, 12))
    plt.subplot(2, 2, 1), plt.title('original_image'), plt.imshow(frame)
    plt.subplot(2, 2, 2), plt.imshow(blue_mask, cmap='gray')
    plt.subplot(2, 2, 3), plt.imshow(green_mask, cmap='gray')
    plt.subplot(2, 2, 4), plt.imshow(red_mask, cmap='gray')

    plt.figure(figsize=(14, 12))
    plt.subplot(2, 2, 1), plt.imshow(blue_res)
    plt.subplot(2, 2, 2), plt.imshow(green_res)
    plt.subplot(2, 2, 3), plt.imshow(red_res)
    plt.subplot(2, 2, 4), plt.imshow(res)
    plt.show()'''
    result = '/disk/sdc/camdata/erzhitu_epoch46' + '/' + image[i]
    cv2.imwrite(result, blue_mask)
#frame = cv2.imread("E:\\data\\02_80.jpg")




