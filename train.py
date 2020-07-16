import argparse
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import data_prep_util

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU: 0].')
parser.add_argument('--model', default='model_fmp', help='Model name [default: model_fmp]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: ./log]')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to train [default: 250]')
parser.add_argument('--batch_size', type=int, default=50, help='Batch Size during training [default: 50]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial momentum rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.8, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

BN_INIT_DECAY = 0.5
BN_DECAY_RATE = 0.5
BN_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.90

DATA_DIR = 'data'
CKPT_DIR = 'ckpt'

HOSTNAME = socket.gethostname()

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE,LOG_DIR)) # backup of model def
os.system('cp train.py %s' % (LOG_DIR)) # backup of train procedure

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(BASE_LEARNING_RATE,
                                               batch * BATCH_SIZE,
                                               DECAY_STEP,
                                               DECAY_RATE,
                                               staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(BN_INIT_DECAY,
                                             batch*BATCH_SIZE,
                                             BN_DECAY_STEP,
                                             BN_DECAY_RATE,
                                             staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1-bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            images_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)

            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            pred = MODEL.get_model(images_pl, is_training_pl, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.argmax(labels_pl,1))
            accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))/float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate',learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            # Add ops to save and restore all variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl:True})

        ops = {
            'images_pl': images_pl,
            'labels_pl': labels_pl,
            'is_training_pl': is_training_pl,
            'pred': pred,
            'loss': loss,
            'train_op': train_op,
            'merged': merged,
            'step': batch,
            'accuracy': accuracy
        }

        for epoch in range(MAX_EPOCH):
            log_string('***** EPOCH %03d *****' % (epoch+1))
            sys.stdout.flush()
            train_one_epoch(sess, ops, train_writer, epoch)
            eval_one_epoch(sess, ops, test_writer)

            # Set point to save variables to disk.
            if (epoch+1) % 10 == 0:
                save_path = saver.save(sess, os.path.join(CKPT_DIR, 'model.ckpt'),global_step=epoch+1)
                log_string("Model saved in file: %s" % save_path)

def train_one_epoch(sess, ops, train_writer, epoch):
    is_training = True

    if(epoch < MAX_EPOCH-10):

        epoch_loss = 0
        epoch_accuracy = 0
        total_batches = 0
        for index in range(1,6):

            Xtrain, Ytrain = data_prep_util.load_aug_CIFAR_train_data(DATA_DIR, index)

            num_batches = int(Xtrain.shape[0]//BATCH_SIZE)
            total_batches += num_batches
            for batch_idx in range(num_batches):
                print('**epoch: %3d, file_index: %d, batch index: %5d / %5d ***' % (epoch, index, batch_idx+1, num_batches))
                batch_x = Xtrain[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE]
                batch_y = Ytrain[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE]
                feed_dict = {
                    ops['images_pl']: batch_x,
                    ops['labels_pl']: batch_y,
                    ops['is_training_pl']: is_training,
                }

                summary, step, _, batch_loss, batch_acc = sess.run([ops['merged'], ops['step'],
                                                                ops['train_op'],ops['loss'],
                                                                ops['accuracy']],feed_dict=feed_dict)

                train_writer.add_summary(summary, step)
                epoch_loss += batch_loss
                epoch_accuracy += batch_acc

    elif(epoch>=MAX_EPOCH-10 & epoch<MAX_EPOCH):
        epoch_loss = 0
        epoch_accuracy = 0
        total_batches = 0
        Xtrain, Ytrain = data_prep_util.load_orig_CIFAR_train_data(DATA_DIR)
        num_batches = int(Xtrain.shape[0] // BATCH_SIZE)
        total_batches += num_batches

        for batch_idx in range(num_batches):
            print('**epoch:%3d, orig batch, batch index: %5d / %d ***' % (epoch,batch_idx+1, num_batches))
            batch_x = Xtrain[batch_idx * BATCH_SIZE:(batch_idx + 1) * BATCH_SIZE]
            batch_y = Ytrain[batch_idx * BATCH_SIZE:(batch_idx + 1) * BATCH_SIZE]

            feed_dict = {
                ops['images_pl']: batch_x,
                ops['labels_pl']: batch_y,
                ops['is_training_pl']: is_training,
            }

            summary, step, _, batch_loss, batch_acc = sess.run([ops['merged'], ops['step'],
                                                                ops['train_op'], ops['loss'],
                                                                ops['accuracy']], feed_dict=feed_dict)
            train_writer.add_summary(summary, step)
            epoch_loss += batch_loss
            epoch_accuracy += batch_acc


    print('epoch: %f, mean loss: %f' % (epoch, epoch_loss / float(total_batches)))
    print('epoch: %f, accuracy: %f' % (epoch, epoch_accuracy / float(total_batches)))

def eval_one_epoch(sess, ops, test_writer):
    is_training = False
    total_accuracy = 0
    Xtest, Ytest = data_prep_util.load_orig_CIFAR_test_data(DATA_DIR)
    total_batches = int(Xtest.shape[0] // BATCH_SIZE)
    for batch_idx in range(total_batches):
        batch_x = Xtest[batch_idx * BATCH_SIZE:(batch_idx+1) * BATCH_SIZE]
        batch_y = Ytest[batch_idx * BATCH_SIZE:(batch_idx+1) * BATCH_SIZE]
        feed_dict = {
            ops['image_pl']: batch_x,
            ops['labels_pl']: batch_y,
            ops['is_training']: is_training
        }

        batch_acc = sess.run(ops['accuracy'],feed_dict=feed_dict)
        total_accuracy += batch_acc
    print('eval mean loss: %f' % (total_accuracy/float(total_batches)))

if __name__ == '__main__':
    train()
    LOG_FOUT.close()