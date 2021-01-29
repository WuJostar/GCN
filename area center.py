'''
此程式碼為論文中自己設計的檢驗方法
'''

import os
import csv
from PIL import Image
import numpy as np

#file_dir = '/home/fan/test2/實驗15 批次訓練結果/實驗15 批次5 batch size 32 訓練10000'
file_dir = '/home/fan/test2/實驗16只有Generator批次訓練結果/實驗16只有Generator批次訓練結果 批次5'

def load_image(file):

    data_path = file
    data_list = os.listdir(data_path)
    result_data_list = []
    answer_data_list = []

    for data_name in data_list:
        if '對應圖' in data_name:
            answer_data_list.append(data_path + '/' + data_name)
        if 'Generator' in file:
            if 'Generator' in data_name:
                result_data_list.append(data_path + '/' + data_name )
        else:
            if 'Autoencoder產生結果(輸入為G(Xa))' in data_name:
                result_data_list.append(data_path + '/' + data_name )
    answer_data_list.sort()
    result_data_list.sort()
    return answer_data_list, result_data_list

def cal_center(img_path):

    img_h = 128
    img_w = 128
    img = Image.open(img_path)
    #img = img('L')
    data = img.getdata()
    data = np.array(data, dtype=np.int32)
    data = np.reshape(data, (img_h, img_w))
    flag = 1
    global top
    global bot
    global right
    global left
    for i in range(img_h):
        for j in range(img_w):
            if data[i,j]>253:
                #print(i,j)
                top = i         
                flag = 0
                break
            if i == img_h-1 and j == img_w-1:
                top = 0
        if flag == 0:
            break
    flag = 1
    for i in range(img_w):
        for j in range(img_h):
            if data[j,i]>253:
                #print(i,j)
                left = i         
                flag = 0
                break
            if i == img_h-1 and j == img_w-1: 
                left = 0
        if flag == 0:
            break
    flag = 1
    for i in range(img_h-1,0,-1):
        for j in range(img_w-1,0,-1):
            if data[i,j] > 253:
                #print(i,j)
                bot = i 
                flag = 0
                break
            if i == 0 and j == 0:
                bot = 0    
        if flag == 0:
            break        
    flag=1
    for i in range(img_w-1,0,-1):
        for j in range(img_h-1,0,-1):
            if data[j,i] > 253:
                #print(i,j)
                right = i
                flag = 0
                break
            if i == 0 and j == 0:
                right = 0
        if flag == 0:
            break
    return round(left+((right-left)/2)), round(top+((bot-top)/2)), (right-left)*(bot-top)

a_list, r_list = load_image(file_dir)
a_x = []
a_y = []
a_area = []
r_x = []
r_y = []
r_area = []
center_gap = []
area_gap = []
for i in np.arange(len(a_list)):
    x_t1, y_t1, area_t1 = cal_center(a_list[i])
    x_t2, y_t2, area_t2 = cal_center(r_list[i])
    a_x.append(x_t1)
    a_y.append(y_t1)
    a_area.append(area_t1)
    r_x.append(x_t2)
    r_y.append(y_t2)
    r_area.append(area_t2)

    center_gap.append(np.sqrt(pow(x_t1-x_t2, 2)+pow(y_t1-y_t2, 2)))
    area_gap.append(np.abs(area_t1-area_t2))

center_gap = np.array(center_gap).reshape(-1, 1)
area_gap = np.array(area_gap).reshape(-1, 1)

f = open(file_dir + "/center_gap.csv","w",newline='')
w = csv.writer(f)
w.writerows(center_gap.astype(np.float32))
f.close()

f = open(file_dir + "/area_gap.csv","w",newline='')
w = csv.writer(f)
w.writerows(area_gap.astype(np.float32))
f.close()
