import os
import numpy as np
from keras.preprocessing import image
import cv2
from PIL import Image
from os import path
# from svmutil import *
import features as f
import copy
import random as r

def return_list(data_path, data_type):
    file_list = [file for file in os.listdir(data_path) if file.lower().endswith(data_type)]
    # print(str(len(file_list)))
    return file_list

data_type = '.jpeg'
data_img_path = './'

file_test_list = return_list(data_img_path, data_type)

f128 = []
f64 = []

dcganImgs=[]

for i in range(2):
    temp_txt = file_test_list[i]
    org_img = cv2.imread(data_img_path + temp_txt)
    # org_img = np.asarray(image.load_img(data_img_path + temp_txt))
    print('Extracting features for Img-{idx} : {temp_txt}'.format(idx=i + 1, temp_txt=temp_txt))
    for i in range(8):
        for j in range(8):
            dcganImgs.append(org_img[i*64:i*64+64,j*64:j*64+64,:])

for i in range(len(dcganImgs)):
    org_img = dcganImgs[i]
    org_img1 = copy.deepcopy(org_img)
    org_img = cv2.resize(org_img, (128, 128))
    org_img1 = cv2.resize(org_img1, (64, 64))
    org_img = np.asarray(org_img)
    org_img1 = np.asarray(org_img1)
    f128.append(f.get_features(org_img))
    f64.append(f.get_features(org_img1))

f128 = np.asarray(f128)
f64 = np.asarray(f64)
np.save('f128x128_DCGAN',f128)
np.save('f64x64_DCGAN',f64)
