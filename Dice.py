'''
此程式碼是在計算Dice係數
'''

import os
import csv
from PIL import Image
import numpy as np

global ans_num
global res_num
global both_num
ans_num = 0
res_num = 0
both_num = 0
#file_dir = '/home/fan/test2/實驗15 批次訓練結果/alpha=0.75/實驗15 批次5 batch size 32 訓練10000'
file_dir = '/home/irisubuntu/林奕帆/實驗1 批次訓練結果/alpha=0.5/實驗1 批次1 batch size 32 訓練10000'

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

def get_dice(ans_path, res_path):

    img_h = 128
    img_w = 128
    ans = Image.open(ans_path)
    res = Image.open(res_path)
    #img = img('L')
    ans_data = ans.getdata()
    ans_data = np.array(ans_data, dtype=np.int32)
    ans_data = np.reshape(ans_data, (img_h, img_w))
    res_data = res.getdata()
    res_data = np.array(res_data, dtype=np.int32)
    res_data = np.reshape(res_data, (img_h, img_w))
    for i in range(img_h):
        for j in range(img_w):
            if ans_data[i, j] > 200:
                global ans_num
                ans_num = ans_num + 1
            if res_data[i, j] > 200:
                global res_num
                res_num = res_num + 1
            if ans_data[i, j] > 200 and res_data[i, j] > 200:
                global both_num
                both_num = both_num + 1
    return ans_num, res_num, both_num, 2*(both_num)/(ans_num + res_num)

a_list, r_list = load_image(file_dir)
iou = []
ans_list = []
res_list = []
both_list = []
for i in np.arange(len(a_list)):
    ans_t, res_t, both_t, iou_t = get_dice(a_list[i], r_list[i])
    iou.append(iou_t)
    ans_list.append(ans_t)
    res_list.append(res_t)
    both_list.append(both_t)
iou = iou + ans_list + res_list + both_list
iou = np.array(iou).reshape([4, 100]).T
f = open(file_dir + "/dice_wc5.csv","w",newline='')
w = csv.writer(f)
w.writerows(iou.astype(np.float32))
f.close()
