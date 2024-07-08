import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
def guiyihua(image,result_path,name):
    print(image)
    result = np.zeros(image.shape, dtype=np.float32)
    # cv2.normalize(image, result, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    result = image / 255.0
    # result = image/127.5 - 1
    print(result)  # float64 (372,419,3) [[[0.98823529 0.96470588 0.91372549] [0.98823529 0.96470588 0.91372549].....]]

    img = np.uint8(result * 255.0)  #
    print((image == img).all())  # true
    plt.imshow(result)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(result_path + name, bbox_inches='tight', pad_inches=0)



img_path="/disk/sdc/AMatrix/data/noill/1"
result_path="/disk/sdc/AMatrix/data/noill/gy/"
image = os.listdir(img_path)
img_tests_path = []
for i in range(len(image)):
    img_tests_path.append(img_path + '/' + image[i])
for k in range(len(img_tests_path)):
    img = cv2.imread(img_tests_path[k])
    guiyihua(img,result_path,str(image[k]))

