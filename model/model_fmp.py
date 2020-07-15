import tensorflow as tf
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR,'../utils'))

import  tf_util

def placeholder_inputs(batch_size):
    Xplaceholder = tf.placeholder(tf.float32, shape=(batch_size,32,32,3))
    Yplaceholder = tf.placeholder(tf.float32, shape=(batch_size,10))
    return Xplaceholder, Yplaceholder


# model : (160nC2 - FMP(cubic root(2))) * 12 - C2 - C1 - output
def get_model(flowing_tensor, is_training, bn_decay=None):
    batch_size = flowing_tensor.get_shape()[0].value


    for i in range(1,11):
        conv_scope = 'conv%d'%(i)
        flowing_tensor =  tf_util.conv2d(flowing_tensor,160*i,[3,3],
                                         stride=[1,1],
                                         padding='SAME',
                                         bn=True,
                                         bn_decay=bn_decay,
                                         is_training=is_training,
                                         scope=conv_scope,)

        fmax_scope = 'fmax_scope%d' % (i)
        # according to paper 'Fractional max-pooling',the arthor set ratio=cubic root(2) nearly 1.2599
        fmax_object = tf_util.fmax_pool2d(flowing_tensor,pooling_ratio=1.26,
                                             pseudo_random=True,
                                             overlapping=True,
                                             scope=fmax_scope)

        flowing_tensor = fmax_object.output

    flowing_tensor = tf.reshape(flowing_tensor, [batch_size, -1])

    flowing_tensor = tf_util.fully_connected(flowing_tensor,512,
                                             bn=True,
                                             bn_decay=bn_decay,
                                             is_training=is_training,
                                             scope='fc1', )

    flowing_tensor = tf_util.fully_connected(flowing_tensor, 128,
                                             bn=True,
                                             bn_decay=bn_decay,
                                             is_training=is_training,
                                             scope='fc2', )

    flowing_tensor = tf_util.fully_connected(flowing_tensor, 10,
                                             bn=True,
                                             bn_decay=bn_decay,
                                             is_training=is_training,
                                             scope='fc3', )

    return flowing_tensor

def get_loss(forward, labels):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=forward,labels=labels))
    return loss

if __name__ == '__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((4,32,32,3))
        outputs = get_model(inputs,tf.constant(True))
        print(outputs)