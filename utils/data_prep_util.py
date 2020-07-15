import os
import sys
import numpy as np
import pickle as p
from sklearn.preprocessing import OneHotEncoder

data_dir = '../data/'
aug_Xtrain_files = ['images_1_1.npy','images_1_2.npy',
                    'images_2_1.npy','images_2_2.npy',
                    'images_3_1.npy','images_3_2.npy',
                    'images_4_1.npy','images_4_2.npy',
                    'images_5_1.npy','images_5_2.npy',]
aug_Ytrain_files = ['labels_1_1.npy','labels_1_2',
                    'labels_2_1.npy','labels_2_2',
                    'labels_3_1.npy','labels_3_2',
                    'labels_4_1.npy','labels_4_2',
                    'labels_5_1.npy','labels_5_2',]

'''
    for original cifar-10
'''

def load_orig_CIFAR_batch(filename):
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

        # 将图像数据标准化
        images = images.astype('float32') / 255.0
        # 将标签数据转成独热编码
        encoder = OneHotEncoder(sparse=False)
        yy = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
        encoder.fit(yy)
        labels = labels.reshape(-1, 1)
        labels = encoder.transform(labels)

        return images, labels


def load_orig_CIFAR_train_data(data_dir):
    """load CIFAR data"""

    images_train = []
    labels_train = []
    for i in range(5):
        f = os.path.join(data_dir, 'data_batch_%d' % (i + 1))
        print('loading ', f)
        # 调用 load_CIFAR_batch( )获得批量的图像及其对应的标签
        image_batch, label_batch = load_orig_CIFAR_batch(f)
        images_train.append(image_batch)
        labels_train.append(label_batch)
        Xtrain = np.concatenate(images_train)
        Ytrain = np.concatenate(labels_train)
        del image_batch, label_batch

    print('finished loadding CIFAR-10 data')

    # 返回训练集的图像和标签，测试集的图像和标签 Xtrain[50000,32,32,3],Ytrain[50000,10]
    return Xtrain, Ytrain

'''
    for loading augmentation training data
'''
def load_aug_CIFAR_train_data(data_dir,index):
    # load augmented X data for training
    Xpart1 = np.load(os.path.join(data_dir,'images_%d_1.npy' % (index)))
    Xpart2 = np.load(os.path.join(data_dir,'images_%d_2.npy' % (index)))
    Xtrain = np.concatenate([Xpart1, Xpart2])
    del Xpart1,Xpart2

    # set a onehot encoder
    encoder = OneHotEncoder(sparse=False)
    yy = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
    encoder.fit(yy)

    # load Y data for training and encode them to onehot format
    Ypart1 = np.load(os.path.join(data_dir, 'labels_%d_1.npy' % (index)))
    Ypart1 = Ypart1.reshape(-1, 1)
    Ypart1 = encoder.transform(Ypart1)
    Ypart2 = np.load(os.path.join(data_dir, 'labels_%d_2.npy' % (index)))
    Ypart2 = Ypart2.reshape(-1, 1)
    Ypart2 = encoder.transform(Ypart2)
    Ytrain = np.concatenate([Ypart1, Ypart2])

    # Xtrain[50000,32,32,3],Ytrain[50000,10]
    return Xtrain, Ytrain

'''
    for loading test data
'''
def load_orig_CIFAR_test_data(data_dir):
    Xtest, Ytest = load_orig_CIFAR_batch(os.path.join(data_dir,'test_batch'))
    # Xtest[10000,32,32,3],Ytest[10000,10]
    return Xtest, Ytest

