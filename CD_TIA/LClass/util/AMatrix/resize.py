import os

import numpy as np
import cv2 as cv

img_path="/disk/sdc/AMatrix/data/noill/1"
result_path="/disk/sdc/AMatrix/data/ill/cj_addweight/"
image = os.listdir(img_path)
img_tests_path = []
for i in range(len(image)):
    img_tests_path.append(img_path + '/' + image[i])
for k in range(len(img_tests_path)):
    img = cv.imread(img_tests_path[k])
    dst = cv.resize(img, (512, 512))
    '''cv.imshow("dst: %d x %d" % (dst.shape[0], dst.shape[1]), dst)
    cv.waitKey(0)
    cv.destroyAllWindows()'''
    uncrop_img_save_path = result_path + str(image[k]) 
    cv.imwrite(uncrop_img_save_path, dst)


