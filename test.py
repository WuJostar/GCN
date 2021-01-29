'''
此程式碼是用於想在已訓練且保存好的模型上測試
'''
import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl
import matplotlib.pyplot as plt
import send_test_data

INPUT_IMAGE_LEN = 128
U_SIZE = 1024
TEST_NUM = 100 #資料夾內的圖片有來自訓練集和測試集的，該常數為來自測試集的數目，請自行設定
IMAGE_CUT_LEN = 580

test_file1_dir  = '/home/irisubuntu/林奕帆/all/test_crop_O' #資料夾內的名稱數字很大的為測試集圖片，其餘為訓練集圖片，測試集圖片數量有更動請同步更改TEST_NUM
test_file2_dir  = '/home/irisubuntu/林奕帆/all/test_crop_G'

#test_file1_dir  = '/home/irisubuntu/林奕帆/test2/大腸資料/第二次訓練資料/ori'
#test_file2_dir  = '/home/irisubuntu/林奕帆/test2/大腸資料/第二次訓練資料/gro'

real_file_dir = '/home/irisubuntu/林奕帆/大腸資料/老師給的樣本/腸鏡數據positive'
check_point_dir = '/home/irisubuntu/林奕帆/實驗1 批次訓練結果/alpha=0.5/實驗1 批次1 batch size 32 訓練10000/model.ckpt'
test = '第3次訓練資料'
real = '真槍實彈2'

class GAN(object):
    def __init__(self):

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        #有ground truth的資料
        self.test_image, self.test_correspondence, self.tatal_num = send_test_data.get_test_data(test_file1_dir, test_file2_dir, INPUT_IMAGE_LEN)
        self.test_result = self.G_net_a(self.G_net_g(self.test_image))
        self.aed = self.D_net_de(self.D_net_en(self.test_correspondence, False), False)
        self.aeg = self.D_net_de(self.D_net_en(self.test_result, True), True)

        #沒有ground truth的資料
        self.real_image, self.real_tatal_num = send_test_data.get_real_data(real_file_dir, INPUT_IMAGE_LEN, IMAGE_CUT_LEN)
        self.real_gen = self.G_net_a(self.G_net_g(self.real_image, True), True)
        self.real_dis = self.D_net_de(self.D_net_en(self.real_gen, True), True)

        self.saver = tf.train.Saver()
        
    def G_net_g(self, z, reuse=False):

        with tf.variable_scope("G_net_g") as scope:
            if reuse:
                scope.reuse_variables()
            #bs = tf.shape(z)[0]
            #print(z.shape)
            conv1 = tc.layers.convolution2d(
                z, 16, [8, 8], [2, 2], 'valid',
                weights_initializer=tcl.xavier_initializer(),#tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
                )
            conv1 = tf.nn.leaky_relu(conv1) 

            #print(conv1.shape)

            conv2 = tc.layers.convolution2d(
                conv1, 32, [5, 5], [2, 2] , 'valid',
                weights_initializer=tcl.xavier_initializer(),
                activation_fn=tf.identity
            )
            conv2 = tf.nn.leaky_relu(conv2) 
            #print(conv2.shape)
            conv2 = tcl.flatten(conv2) 
            #print(conv2.shape)

            fc1 = tc.layers.fully_connected(
                conv2, 2048, #3650,
                weights_initializer=tcl.xavier_initializer(),
                activation_fn=tf.identity
            )
            '''
            fc1 = self.batch_normalization(fc1, self.is_training)
            fc1 = tf.nn.leaky_relu(fc1)

            fc1 = tc.layers.fully_connected(
                conv2, 2048, #3650,
                weights_initializer=tcl.xavier_initializer(),
                activation_fn=tf.identity
            )
            #添加層
            '''
            fc1 = self.batch_normalization(fc1)
            fc1 = tf.nn.leaky_relu(fc1)
            fc2 = tc.layers.fully_connected(fc1, U_SIZE, activation_fn=tf.identity)
            return fc2
                        
    def G_net_a(self, u, reuse=False):

        with tf.variable_scope("G_net_a") as scope:
            if reuse:
                scope.reuse_variables()
            #print(u.shape)
            bs = tf.shape(u)[0]
            '''
            #添加層
            fc1 = tc.layers.fully_connected(
                u, 2048,
                weights_initializer=tcl.xavier_initializer(),
                #weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            fc1 = self.batch_normalization(fc1, self.is_training)
            fc1 = tf.nn.leaky_relu(fc1)
            '''
            fc1 = tc.layers.fully_connected(
                u, 2048,
                weights_initializer=tcl.xavier_initializer(),
                #weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            fc1 = self.batch_normalization(fc1)
            fc1 = tf.nn.leaky_relu(fc1)
            #print(fc1.shape)
            fc2 = tc.layers.fully_connected(
                fc1, 29*29*32,#15 * 15 * 32,
                weights_initializer=tcl.xavier_initializer(),
                #weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            #print(fc2.shape)
            fc2 = tf.reshape(fc2, tf.stack([bs, 29, 29, 32]))
            #print(fc2.shape)
            fc2 = self.batch_normalization(fc2)
            fc2 = tf.nn.leaky_relu(fc2)

            conv1 = tc.layers.convolution2d_transpose(
                fc2, 16, [5, 5], [2, 2], 'valid',
                weights_initializer=tcl.xavier_initializer(),
                #weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            #print(conv1.shape)

            #conv1 = self.batch_normalization(conv1, self.is_training)
            conv1 = tf.nn.leaky_relu(conv1)
            conv2 = tc.layers.convolution2d_transpose(
                conv1, 1, [8, 8], [2, 2], 'valid',
                weights_initializer=tcl.xavier_initializer(),
                #weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.sigmoid
            )
            #print(conv2.shape)
            #conv2 = tf.reshape(conv2, tf.stack([bs, 4096]))
            return conv2
       
    def D_net_en(self, x, reuse=False):

       with tf.variable_scope("D_net_en") as scope:
            if reuse:
                scope.reuse_variables()

            #bs = tf.shape(z)[0]
            
            conv1 = tc.layers.convolution2d(
                x, 16, [8, 8], [2, 2], 'valid',
                weights_initializer=tcl.xavier_initializer(),
                activation_fn=tf.identity)

            conv1 = tf.nn.leaky_relu(conv1)

            conv2 = tc.layers.convolution2d(
                conv1, 32, [5, 5], [2, 2], 'valid',
                weights_initializer=tcl.xavier_initializer(),
                activation_fn=tf.identity
            )
            conv2 = tf.nn.leaky_relu(conv2)
            conv2 = tcl.flatten(conv2)

            fc1 = tc.layers.fully_connected(
                conv2, 2048,
                weights_initializer=tcl.xavier_initializer(),
                activation_fn=tf.identity
            )
            fc1 = self.batch_normalization(fc1)
            fc1 = tf.nn.leaky_relu(fc1)
            fc2 = tc.layers.fully_connected(fc1, U_SIZE, activation_fn=tf.identity)
            return fc2

    def D_net_de(self, u, reuse=False):

        with tf.variable_scope("D_net_de") as scope:
            if reuse:
                scope.reuse_variables()

            bs = tf.shape(u)[0]

            fc1 = tc.layers.fully_connected(
                u, 2048,
                weights_initializer=tcl.xavier_initializer(),
                #weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            fc1 = self.batch_normalization(fc1)
            fc1 = tf.nn.leaky_relu(fc1)

            fc2 = tc.layers.fully_connected(
                fc1, 29*29*32,#15 * 15 * 32,
                weights_initializer=tcl.xavier_initializer(),
                #weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            fc2 = tf.reshape(fc2, tf.stack([bs, 29, 29, 32]))
            fc2 = self.batch_normalization(fc2)
            fc2 = tf.nn.leaky_relu(fc2)

            conv1 = tc.layers.convolution2d_transpose(
                fc2, 16, [5, 5], [2, 2], 'valid',
                weights_initializer=tcl.xavier_initializer(),
                #weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            conv1 = self.batch_normalization(conv1)
            conv1 = tf.nn.leaky_relu(conv1)
            conv2 = tc.layers.convolution2d_transpose(
                conv1, 1, [8, 8], [2, 2], 'valid',
                weights_initializer=tcl.xavier_initializer(),
                #weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.sigmoid
            )
            #conv2 = tf.reshape(conv2, tf.stack([bs, 4096]))
            return conv2
    
    def batch_normalization(self, input):
        output = tf.layers.batch_normalization(input, training=False)
        return output
    
    def test(self):
        self.sess.run(tf.global_variables_initializer())
        self.saver.restore(self.sess, check_point_dir)
        
        #測試沒有ground truth的資料
        result_real_image = self.sess.run(self.real_image)
        result_real_gen = self.sess.run(self.real_gen)
        result_real_dis = self.sess.run(self.real_dis)
        for i in np.arange(self.real_tatal_num):
            images_encode1 = tf.image.encode_jpeg(result_real_image[i, :, :, :]*255)
            images_encode2 = tf.image.encode_jpeg(result_real_gen[i, :, :, :]*255)
            images_encode3 = tf.image.encode_jpeg(result_real_dis[i, :, :, :]*255)
            fname1 = tf.constant('/home/irisubuntu/林奕帆/test2/' + real + '/%d.原圖.jpeg' %(i+1))
            fname2 = tf.constant('/home/irisubuntu/林奕帆/test2/' + real + '/%d.中間產物.jpeg' %(i+1))
            fname3 = tf.constant('/home/irisubuntu/林奕帆/test2/' + real + '/%d.最終輸出.jpeg' %(i+1))
            fwrite1 = tf.write_file(fname1, images_encode1)
            fwrite2 = tf.write_file(fname2, images_encode2)
            fwrite3 = tf.write_file(fname3, images_encode3)
            result1 = self.sess.run(fwrite1)
            #print(type(result_real_dis))
            result2 = self.sess.run(fwrite2)
            result3 = self.sess.run(fwrite3)
        
        
        #測試有ground truth的測試集的資料
        result_image = self.sess.run(self.test_result)
        result_correspondence = self.sess.run(self.test_correspondence)
        result_ori = self.sess.run(self.test_image)
        result_aed = self.sess.run(self.aed)
        result_aeg = self.sess.run(self.aeg)
        for i in np.arange(self.tatal_num):       
            images_encode3 = tf.image.encode_jpeg(result_ori[i, :, :, :]*255)
            images_encode2 = tf.image.encode_jpeg(result_correspondence[i, :, :, :]*255)
            images_encode1 = tf.image.encode_jpeg(result_image[i, :, :, :]*255)
            images_encode4 = tf.image.encode_jpeg(result_aed[i, :, :, :]*255)
            images_encode5 = tf.image.encode_jpeg(result_aeg[i, :, :, :]*255)
            if i < self.tatal_num-TEST_NUM:
                fname3 = tf.constant('/home/irisubuntu/林奕帆/' + test + '/訓練集結果/%d.訓練集原圖.jpeg' %(i+1))
                fname2 = tf.constant('/home/irisubuntu/林奕帆/' + test + '/訓練集結果/%d.訓練集對應圖.jpeg' %(i+1))
                fname1 = tf.constant('/home/irisubuntu/林奕帆/' + test + '/訓練集結果/%d.訓練集Generator產生結果.jpeg' %(i+1))
                fname4 = tf.constant('/home/irisubuntu/林奕帆/' + test + '/訓練集結果/%d.訓練集Autoencoder產生結果(輸入為Xd).jpeg' %(i+1))
                fname5 = tf.constant('/home/irisubuntu/林奕帆/' + test + '/訓練集結果/%d.訓練集Autoencoder產生結果(輸入為G(Xa)).jpeg' %(i+1))
            else:
                fname3 = tf.constant('/home/irisubuntu/林奕帆/' + test + '/測試集結果/%d.測試集原圖.jpeg' %(i-self.tatal_num+TEST_NUM+1))
                fname2 = tf.constant('/home/irisubuntu/林奕帆/' + test + '/測試集結果/%d.測試集對應圖.jpeg' %(i-self.tatal_num+TEST_NUM+1))
                fname1 = tf.constant('/home/irisubuntu/林奕帆/' + test + '/測試集結果/%d.測試集Generator產生結果.jpeg' %(i-self.tatal_num+TEST_NUM+1))
                fname4 = tf.constant('/home/irisubuntu/林奕帆/' + test + '/測試集結果/%d.測試集Autoencoder產生結果(輸入為Xd).jpeg' %(i-self.tatal_num+TEST_NUM+1))
                fname5 = tf.constant('/home/irisubuntu/林奕帆/' + test + '/測試集結果/%d.測試集Autoencoder產生結果(輸入為G(Xa)).jpeg' %(i-self.tatal_num+TEST_NUM+1))          
            fwrite3 = tf.write_file(fname3, images_encode3)
            fwrite2 = tf.write_file(fname2, images_encode2)
            fwrite1 = tf.write_file(fname1, images_encode1)
            fwrite4 = tf.write_file(fname4, images_encode4)
            fwrite5 = tf.write_file(fname5, images_encode5)
            result3 = self.sess.run(fwrite3)
            result2 = self.sess.run(fwrite2)
            result1 = self.sess.run(fwrite1)
            result4 = self.sess.run(fwrite4)
            result5 = self.sess.run(fwrite5)
        
        '''
        result_correspondence = self.sess.run(self.test_correspondence)
        #result_aeg = self.sess.run(self.aeg)
        result_image = self.sess.run(self.test_result)
        for i in np.arange(self.tatal_num):
            images_encode2 = tf.image.encode_jpeg(result_correspondence[i, :, :, :]*255)
            #images_encode5 = tf.image.encode_jpeg(result_aeg[i, :, :, :]*255)
            images_encode1 = tf.image.encode_jpeg(result_image[i, :, :, :]*255)

            if i < 9:
                fname2 = tf.constant('/home/irisubuntu/林奕帆/test2/' + test + '/00%d.Xd.jpeg' %(i+1))
                #fname5 = tf.constant('/home/irisubuntu/林奕帆/test2/' + test + '/00%d.G`(Xa).jpeg' %(i+1))
                fname1 = tf.constant('/home/irisubuntu/林奕帆/test2/' + test + '/00%d.G(Xa).jpeg' %(i+1))

            else:
                fname2 = tf.constant('/home/irisubuntu/林奕帆/test2/' + test + '/0%d.Xd.jpeg' %(i+1))
                #fname5 = tf.constant('/home/irisubuntu/林奕帆/test2/' + test + '/0%d.G' + "'" + '(Xa).jpeg' %(i+1))
                fname1 = tf.constant('/home/irisubuntu/林奕帆/test2/' + test + '/0%d.G(Xa).jpeg' %(i+1))
            fwrite2 = tf.write_file(fname2, images_encode2)
            fwrite1 = tf.write_file(fname1, images_encode1)
            #fwrite5 = tf.write_file(fname5, images_encode5)
            result2 = self.sess.run(fwrite2)
            result1 = self.sess.run(fwrite1)
            #result5 = self.sess.run(fwrite5)
        '''
#--------------------------------------------------------------------------------------------------------------------

tf.reset_default_graph()
gan = GAN()
gan.test()
