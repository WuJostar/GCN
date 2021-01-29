'''
論文中的類SegNet
'''
import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl
import pretreat
import send_test_data

BATCH_SIZE = 32
CAPACITY = 2000
INPUT_IMAGE_LEN = 128
U_SIZE = 1024
LEARNING_RATE = 0.0001
TRAIN_EPOCH = 5000
CHECK = 10
TEST_NUM = 0 #資料夾內的圖片有來自訓練集和測試集的，該常數為來自測試集的數目，請自行設定

test = '實驗16只有Generator 批次10'

#file1_dir = '/home/fan/test2/大腸資料/最初少量版本/Original'
#file2_dir = '/home/fan/test2/大腸資料/最初少量版本/Ground Truth'

#file1_dir = '/home/fan/test2/大腸資料/大量版本/全部/Original'
#file2_dir = '/home/fan/test2/大腸資料/大量版本/全部/Ground Truth'

file1_dir = '/home/fan/test2/大腸資料/最終實驗版本/Original'
file2_dir = '/home/fan/test2/大腸資料/最終實驗版本/Ground Truth'

#test_file1_dir  = '/home/fan/test2/大腸資料/測試/Original'
#test_file2_dir  = '/home/fan/test2/大腸資料/測試/Ground Truth'

#test_file1_dir = '/home/fan/test2/大腸資料/大量版本/測試/Original'
#test_file2_dir = '/home/fan/test2/大腸資料/大量版本/測試/Ground Truth'

test_file1_dir = '/home/fan/test2/大腸資料/訓練資料批次/批次10/Origin_10'
test_file2_dir = '/home/fan/test2/大腸資料/訓練資料批次/批次10/Ground Truth_10'

check_point_dir = '/home/fan/test2/' + test + '/model.ckpt'

class GAN(object):
    def __init__(self):

        self.is_training = tf.placeholder(tf.bool)

        self.train, self.train_correspondence = pretreat.get_file_path(file1_dir, file2_dir)
        self.train_batch, self.train_correspondence_batch = pretreat.get_batch(self.train, self.train_correspondence,
                                                            INPUT_IMAGE_LEN, BATCH_SIZE ,CAPACITY)

        self.g_u = self.G_net_g(self.train_batch, False)
        self.fake = self.G_net_a(self.g_u, False)

        self.test_image, self.test_correspondence, self.tatal_num = send_test_data.get_test_data(test_file1_dir, test_file2_dir, INPUT_IMAGE_LEN)
        self.test_result = self.G_net_a(self.G_net_g(self.test_image, True), True)
      
        self.G_loss = tf.losses.mean_squared_error(self.fake, self.train_correspondence_batch)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)
        
        t_vars = tf.trainable_variables()
        self.G_vars = [var for var in t_vars if 'G_net' in var.name]

        g_list = tf.global_variables()
        self.G_bn_moving_mean_vars = [g for g in g_list if 'moving_mean' in g.name]
        self.G_bn_moving_mean_vars = [g for g in self.G_bn_moving_mean_vars if 'G_net' in g.name]

        self.G_bn_moving_variance_vars = [g for g in g_list if 'moving_variance' in g.name]
        self.G_bn_moving_variance_vars = [g for g in self.G_bn_moving_variance_vars if 'G_net' in g.name]

        self.G_vars += self.G_bn_moving_mean_vars
        self.G_vars += self.G_bn_moving_variance_vars

        self.vars = self.G_vars
      
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.G_train = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(self.G_loss, var_list=self.G_vars)
         
        self.saver = tf.train.Saver(self.vars)
        
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
            fc1 = self.batch_normalization(fc1, self.is_training)
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
            fc1 = self.batch_normalization(fc1, self.is_training)
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
            fc2 = self.batch_normalization(fc2, self.is_training)
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
    
    '''
    def margin(self, dgz):

        zero = tf.constant(0, dtype=tf.float32)
        m = tf.constant(mar, dtype=tf.float32)

        result = tf.cond(zero > tf.subtract(m, dgz), lambda: zero, lambda: tf.subtract(m, dgz))

        return result
    '''
    
    def batch_normalization(self, input, is_training):

        output = tf.layers.batch_normalization(input, training = is_training)

        return output
    
    def training(self):
        
        self.sess.run(tf.global_variables_initializer())

        lossg = []
        count = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
    
        try:

            for step in np.arange(TRAIN_EPOCH+1):

                if coord.should_stop():
                    break

                self.sess.run(self.G_train, feed_dict={self.is_training: True})
                G_loss = self.sess.run(self.G_loss, feed_dict={self.is_training: False})

                if step % CHECK == 0:
                    print("step:%d" % (step))
                    print("G_loss = %f" % (G_loss))
                    if count > 50:
                        lossg.append(G_loss)
                    count = count + 1
                
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')

        finally:

            coord.request_stop()
            coord.join(threads)
            #self.sess.close()
            print('training is finishied')

        #print(self.G_vars)

        print('recoding model and loss')

        save_path = self.saver.save(self.sess, check_point_dir)

        lossg = np.array(lossg).reshape(-1, 1)
        lossg = lossg.astype(np.float32)  
        
        x = np.arange(1,len(lossg)+1)
        y = lossg
        fig = plt.figure()
        ax = plt.gca()
        ax.set_xlabel('epoch')
        ax.set_ylabel('G_loss')
        #plt.legend(loc='upper right')
        ax.plot(x, y, color='r', linewidth=1, alpha=0.6, label='G_loss')
        plt.savefig("/home/fan/test2/"+ test + "/G_loss.png")

        f = open("/home/fan/test2/"+ test + "/G_loss.csv","w",newline='')
        w = csv.writer(f)
        w.writerows(lossg)
        f.close()
    
    def test(self):

        print('generating result data')

        result_image = self.sess.run(self.test_result, feed_dict={self.is_training: False})
        result_correspondence = self.sess.run(self.test_correspondence, feed_dict={self.is_training: False})
        result_ori = self.sess.run(self.test_image, feed_dict={self.is_training: False})

        for i in np.arange(self.tatal_num):
            
            images_encode3 = tf.image.encode_jpeg(result_ori[i, :, :, :]*255)
            images_encode2 = tf.image.encode_jpeg(result_correspondence[i, :, :, :]*255)
            images_encode1 = tf.image.encode_jpeg(result_image[i, :, :, :]*255)
            
            fname3 = tf.constant('/home/fan/test2/' + test + '/%d.原圖.jpeg' %(i+1))
            fname2 = tf.constant('/home/fan/test2/' + test + '/%d.對應圖.jpeg' %(i+1))
            fname1 = tf.constant('/home/fan/test2/' + test + '/%d.Generator產生結果.jpeg' %(i+1))

            fwrite3 = tf.write_file(fname3, images_encode3)
            fwrite2 = tf.write_file(fname2, images_encode2)
            fwrite1 = tf.write_file(fname1, images_encode1)

            result3 = self.sess.run(fwrite3)
            result2 = self.sess.run(fwrite2)
            result1 = self.sess.run(fwrite1)
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------

tf.reset_default_graph()
gan = GAN()
gan.training()
gan.test()










