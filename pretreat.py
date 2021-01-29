'''
此程式碼主要功用如下：
1.對訓練圖片進行裁切縮放等預處理
2.將訓練圖片包成一個個batch

可參考以下網址
https://www.youtube.com/watch?v=y06-jHGPug4
https://www.youtube.com/watch?v=G1X-qJaCfII
https://www.youtube.com/watch?v=cIbYy1hhNlQ&t
'''
import os
import numpy as np
import tensorflow as tf

def get_file_path(ori_file_dir, gro_file_dir):
    origin_name_list = os.listdir(ori_file_dir)
    origin_name_list.sort(key= lambda x:int(x[1:-5]))
    ground_truth_name_list = os.listdir(gro_file_dir)
    ground_truth_name_list.sort(key= lambda x:int(x[1:-5]))
    origin_path = []
    ground_truth_path = []
    for origin_name in origin_name_list:
        origin_path.append(ori_file_dir + '/' + origin_name)
    for ground_truth_name in ground_truth_name_list:
        ground_truth_path.append(gro_file_dir + '/' + ground_truth_name)
    tmp = np.vstack((np.array(origin_path), np.array(ground_truth_path)))
    tmp = tmp.transpose()
    np.random.shuffle(tmp)
    origin_list = list(tmp[:,0])
    ground_truth_list = list(tmp[:,1])
    return origin_list, ground_truth_list

def get_batch(origin_list, ground_truth_list, input_image_len, batch_size, capacity):
    origin_tmp = tf.cast(origin_list, tf.string)
    ground_truth_tmp = tf.cast(ground_truth_list, tf.string)
    input_queue = tf.train.slice_input_producer([origin_tmp, ground_truth_tmp])  
    ori_content = tf.read_file(input_queue[0])  
    gro_content = tf.read_file(input_queue[1])
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
    ori_batch, gro_batch = tf.train.batch([ori_image, gro_image], batch_size=batch_size, num_threads=64, capacity=capacity)
    return ori_batch, gro_batch

'''
#測試用
import matplotlib.pyplot as plt

BATCH_SIZE = 4
CAPACITY = 256
INPUT_IMAGE_LEN = 128
ori_file_dir  = '/home/irisubuntu/林奕帆/all/Original'
gro_file_dir  = '/home/irisubuntu/林奕帆/all/Ground Truth'

ori_list, gro_list = get_file_path(ori_file_dir, gro_file_dir)
ori_batch, gro_batch = get_batch(ori_list, gro_list, INPUT_IMAGE_LEN, BATCH_SIZE, CAPACITY)

with tf.Session() as sess:  
    #i = 0  
    coord = tf.train.Coordinator()  
    threads = tf.train.start_queue_runners(coord=coord)  
    try:  
        #while not coord.should_stop() and i < 5:
            
        ori, gro = sess.run([ori_batch, gro_batch]) 

        for j in np.arange(BATCH_SIZE):
 
            images_encode = tf.image.encode_jpeg(ori[j, :, :, :]*255)
            fname = tf.constant('/home/fan/test2/' + '%d原圖.jpeg' %(j+1))
            fwrite = tf.write_file(fname, images_encode)
            result = sess.run(fwrite)

            images_encode2 = tf.image.encode_jpeg(gro[j, :, :, :]*255)
            fname2 = tf.constant('/home/fan/test2/' + '%d對應.jpeg' %(j+1))
            fwrite2 = tf.write_file(fname2, images_encode2)
            result2 = sess.run(fwrite2)

            #i += 1  
    except tf.errors.OutOfRangeError:  
        print("done!")  
    finally:  
        coord.request_stop()  
    coord.join(threads)
'''
