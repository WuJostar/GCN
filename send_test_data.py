'''
此程式碼主要是將複數測試圖片合成一個多維tensor以供測試
'''
import os
import tensorflow as tf
import numpy as np
 
 #讀取沒有ground truth的資料
def get_real_data(file_dir, input_image_len, image_cut_len):
    data_path = []
    data_list = os.listdir(file_dir)
    data_list.sort()
    #data_list.sort(key= lambda x:int(x[:-4]))
    for data_name in data_list:
        data_path.append(file_dir + '/' + data_name)
    tatal_num = len(data_path)
    for num in np.arange(tatal_num):
        data_path_tmp = tf.reshape(data_path[num], [])
        data_path_tmp = tf.cast(data_path_tmp, tf.string)    
        data_content = tf.read_file(data_path_tmp)
        data_image = tf.image.decode_jpeg(data_content, channels=3)        
        data_image = tf.image.resize_image_with_crop_or_pad(data_image, image_cut_len, image_cut_len) 
        data_image = tf.image.resize_images(data_image, [input_image_len, input_image_len])        
        data_image = tf.cast(data_image, tf.float32)
        data_image = data_image/255        
        data_image = tf.reshape(data_image, [1, input_image_len, input_image_len, 3])
        if num == 0:
            data_image_all = data_image
        else:
            data_image_all = tf.concat([data_image_all, data_image], 0)
    data_image_all = tf.reshape(data_image_all, [tatal_num, input_image_len, input_image_len, 3])
    return data_image_all, tatal_num

#讀取有ground truth的測試集的資料
def get_test_data(ori_file_dir, gro_file_dir, input_image_len):
    origin_path = []
    ground_truth_path = []
    origin_list = os.listdir(ori_file_dir)
    origin_list.sort(key= lambda x:int(x[1:-5]))
    ground_truth_list = os.listdir(gro_file_dir)
    ground_truth_list.sort(key= lambda x:int(x[1:-5]))
    for origin_name in origin_list:
        origin_path.append(ori_file_dir + '/' + origin_name)
    for ground_truth_name in ground_truth_list:
        ground_truth_path.append(gro_file_dir + '/' + ground_truth_name)
    tatal_num = len(origin_path)
    #print(origin_path)
    #print(ground_truth_path)
    for num in np.arange(tatal_num):
        origin_tmp = tf.reshape(origin_path[num], [])
        ground_truth_tmp = tf.reshape(ground_truth_path[num], [])
        origin_tmp = tf.cast(origin_tmp, tf.string)
        ground_truth_tmp = tf.cast(ground_truth_tmp, tf.string)
        ori_content = tf.read_file(origin_tmp)
        gro_content = tf.read_file(ground_truth_tmp)
        ori_image = tf.image.decode_jpeg(ori_content, channels=3)
        gro_image = tf.image.decode_jpeg(gro_content, channels=1)
        ori_image = tf.image.resize_image_with_crop_or_pad(ori_image, 384, 384) #圖片最大邊長為384
        gro_image = tf.image.resize_image_with_crop_or_pad(gro_image, 384, 384)
        ori_image = tf.image.resize_images(ori_image, [input_image_len, input_image_len])
        gro_image = tf.image.resize_images(gro_image, [input_image_len, input_image_len])
        ori_image = tf.cast(ori_image, tf.float32)
        gro_image = tf.cast(gro_image, tf.float32)
        ori_image = ori_image/255
        gro_image = gro_image/255
        #print(ori_image.shape)
        ori_image = tf.reshape(ori_image, [1, input_image_len, input_image_len, 3])
        gro_image = tf.reshape(gro_image, [1, input_image_len, input_image_len, 1])
        if num == 0:
            ori_image_all = ori_image
            gro_image_all = gro_image
        else:
            ori_image_all = tf.concat([ori_image_all, ori_image], 0)
            gro_image_all = tf.concat([gro_image_all, gro_image], 0)
    ori_image_all = tf.reshape(ori_image_all, [tatal_num, input_image_len, input_image_len, 3])
    gro_image_all = tf.reshape(gro_image_all, [tatal_num, input_image_len, input_image_len, 1])
    return ori_image_all, gro_image_all, tatal_num
'''
#測試用
import matplotlib.pyplot as plt
BATCH_SIZE = 1
CAPACITY = 256
INPUT_IMAGE_LEN = 128
ori_file_dir = '/home/fan/test2/大腸資料/測試/Original'
gro_file_dir = '/home/fan/test2/大腸資料/測試/Ground Truth'

ori_batch, gro_batch, tatal_num = get_test_data(ori_file_dir, gro_file_dir,INPUT_IMAGE_LEN)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)

for i in np.arange(tatal_num):
    images_encode3 = tf.image.encode_jpeg(tf.cast(ori_batch[i, :, :, :]*255, tf.uint8))
    fname3 = tf.constant('/home/fan/test2/%d原圖.jpeg' %(i+1))
    fwrite3 = tf.write_file(fname3, images_encode3)
    result3 = sess.run(fwrite3)
    images_encode2 = tf.image.encode_jpeg(tf.cast(gro_batch[i, :, :, :]*255, tf.uint8))
    fname2 = tf.constant('/home/fan/test2/%d對應圖.jpeg' %(i+1)) 
    fwrite2 = tf.write_file(fname2, images_encode2)
    result2 = sess.run(fwrite2)
'''
