
import os
import re
import cv2
import cv2 as cv

def img_to_erzhi(img_path):
    image = os.listdir(img_path)
    img_tests_path = []
    for i in range(len(image)):
        img_tests_path.append(img_path + '/' + image[i])
        img=cv.imread(img_tests_path[i])
        src1 = cv.resize(img, (512, 512))
        gray = cv.cvtColor(src1, cv.COLOR_BGR2GRAY)
        ret, img1 = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)
        result='/disk/sdc/camdata/rr'+'/'+image[i]
        cv.imwrite(result, img1)

def img_add_img(img1_path,img2_path):
    image1 = os.listdir(img1_path)
    image2 = os.listdir(img2_path)
    img1_tests_path = []
    img2_tests_path = []
    for i in range(len(image1)):
        img1_tests_path.append(img1_path + '/' + image1[i])

    for i in range(len(image2)):
        img2_tests_path.append(img2_path + '/' + image2[i])

    for i in range(len(img2_tests_path)):
        img1 = cv.imread(img2_tests_path[i])
        name = image2[i].split('.')[0]
        img1 = cv.resize(img1, (512, 512))
        for j in range(len(image1)):

            if re.match(name, image1[j]):
                picture = cv2.imread(img1_tests_path[j])
                print(name)
                print(image1[j])
                print(i)
                picture = cv.resize(picture, (512, 512))
                #c=img1+picture
                c= cv.addWeighted(img1, 0.8, picture, 0.4, 10)
                #c = cv.add(img1, picture)
                result = '/disk/sdc/AMatrix/data/noill/add' + '/' + image2[i]
                cv.imwrite(result, c)
                break

def img_bitwise_and_img(img1_path,img2_path):
    image1 = os.listdir(img1_path)
    image2 = os.listdir(img2_path)
    img1_tests_path = []
    img2_tests_path = []
    for i in range(len(image1)):
        img1_tests_path.append(img1_path + '/' + image1[i])

    for i in range(len(image2)):
        img2_tests_path.append(img2_path + '/' + image2[i])

    for i in range(len(img1_tests_path)):
        img1 = cv.imread(img1_tests_path[i])
        name = image1[i].split('.')[0]

        for j in range(len(image2)):

            if re.match(name, image2[j]):
                picture = cv2.imread(img2_tests_path[j])
                picture = cv.resize(picture, (512, 512))
                print(name)
                print(image2[j])
                print(i)
                c = cv.bitwise_and(img1, picture)
                result = '/disk/sdc/AMatrix/data/noill/add' + '/' + image1[i]
                cv.imwrite(result, c)
                break

if __name__ == '__main__':

    img1_path = '/disk/sdc/AMatrix/data/noill/1/'
    img2_path='/disk/sdc/AMatrix/data/noill/2/'
    img_add_img(img1_path,img2_path)




