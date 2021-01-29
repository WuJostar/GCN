'''
論文中的白色像素計算
'''

import os
import csv
from PIL import Image
import numpy as np

#file_dir = '/home/fan/test2/實驗15 批次訓練結果/實驗15 批次10 batch size 32 訓練10000'
file_dir = '/home/fan/test2/實驗16只有Generator批次訓練結果/實驗16只有Generator批次訓練結果 批次6'

def load_image(file):

    data_path = file
    data_list = os.listdir(data_path)
    result_data_list = []
    for data_name in data_list:
        #if 'Autoencoder產生結果(輸入為G(Xa))' in data_name:
        if 'Generator' in data_name:
            result_data_list.append(data_path + '/' + data_name )
    result_data_list.sort()
    return result_data_list

def cal_center(img_path):

    img_h = 128
    img_w = 128
    img = Image.open(img_path)
    #img = img('L')
    data = img.getdata()
    data = np.array(data, dtype=np.int32)
    data = np.reshape(data, (img_h, img_w))
    global white
    white = 0
    for i in range(img_h):
        for j in range(img_w):
            if data[i,j]>253:
                white = white + 1
    return white

r_list = load_image(file_dir)
white_point = []
for i in np.arange(len(r_list)):
    white_point.append(cal_center(r_list[i]))
white_point = sum(white_point)
white_point = np.array(white_point).reshape(1, 1)
f = open(file_dir + "/white_point.csv","w",newline='')
w = csv.writer(f)
w.writerows(white_point.astype(np.float32))
f.close()

