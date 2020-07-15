'''
    修改aug_train_filename,aug_test_filename,half_Xtrain,half_Ytrain
'''

import os
import numpy as np
import tensorflow as tf
import pickle as p

index = 5
data_dir = '/home/aragorn/Documents/lecture/Tensorflow-Summer-2020/data/cifar-10-batches-py/'
orig_filename = os.path.join(data_dir,'data_batch_%d' % (index))
aug_train_filename = 'images_%d_2.npy' % (index)
aug_test_filename = 'labels_%d_2.npy' % (index)


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb')as f:
        # 一个样本由标签和图像数据组成
        # <1 x label><3072 x pixel> (3072=32x32x3)
        # ...
        # <1 x label><3072 x pixel>
        data_dict = p.load(f, encoding='bytes')
        images = data_dict[b'data']
        labels = data_dict[b'labels']

        # 把原始数据结构调整为: BCWH
        images = images.reshape(10000, 3, 32, 32)
        # tensorflow处理图像数据的结构：BWHC
        # 把通道数据C移动到最后一个维度
        images = images.transpose(0, 2, 3, 1)

        labels = np.array(labels)

        return images, labels


Xtrain,Ytrain = load_CIFAR_batch(orig_filename)
print('type(Xtrain):',type(Xtrain),'Xtrain.shape',Xtrain.shape)

orig_num = Xtrain.shape[0]
print('orig_num:',orig_num)

# half_Xtrain = Xtrain[0:orig_num//2]
# half_Ytrain = Ytrain[0:orig_num//2]
half_Xtrain = Xtrain[orig_num//2:orig_num]
half_Ytrain = Ytrain[orig_num//2:orig_num]
print('half_Xtrain.shape:',half_Xtrain.shape,'half_Ytrain.shape:',half_Xtrain.shape)

# 删除变量 Xtrain
del Xtrain

half_Xtrain = tf.image.convert_image_dtype(half_Xtrain,tf.float32)

sess = tf.Session()
for i in range(orig_num//2):
    # 左右翻转
    a = tf.image.flip_left_right(half_Xtrain[i])
    a = tf.reshape(a, [1, 32, 32, 3])
    half_Xtrain = tf.concat([half_Xtrain, a], 0)
    # 调整饱和度
    a = tf.image.adjust_saturation(half_Xtrain[i],3)
    a = tf.reshape(a, [1, 32, 32, 3])
    half_Xtrain = tf.concat([half_Xtrain, a], 0)
    # 调整亮度
    a = tf.image.adjust_brightness(half_Xtrain[i],0.4)
    a = tf.reshape(a, [1, 32, 32, 3])
    half_Xtrain = tf.concat([half_Xtrain, a], 0)
    # 旋转90度
    a = tf.image.rot90(half_Xtrain[i])
    a = tf.reshape(a, [1, 32, 32, 3])
    half_Xtrain = tf.concat([half_Xtrain, a], 0)
    # 标签数据追加
    half_Ytrain = np.append(half_Ytrain, [half_Ytrain[i],half_Ytrain[i],half_Ytrain[i],half_Ytrain[i]])
    if (i+1)%50 == 0:
        print(i+1,half_Xtrain,half_Ytrain.shape)

print('start transfer to np')
np_half_Xtrain = sess.run(half_Xtrain)
print('transfer finished')
sess.close()
# 删除变量
del half_Xtrain

print('type(Xtrain):',type(np_half_Xtrain),'Xtrain.shape',np_half_Xtrain.shape)
print('type(Ytrain):',type(half_Ytrain),'Ytrain.shape',half_Ytrain.shape)

# 保存Xtrain, Ytrain 数组
np.save(aug_train_filename,np_half_Xtrain)
np.save(aug_test_filename,half_Ytrain)

print('part1 saved!')