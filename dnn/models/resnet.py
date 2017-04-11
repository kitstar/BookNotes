import argparse
import sys

import numpy as np
import tensorflow as tf
import utils
import tensorflow.contrib.slim as slim
import resnet_utils
from utils import real_type
from tensorflow.contrib.slim.nets import resnet_v1
# from tensorflow.contrib.slim.python.slim.nets import resnet_v1


class KitModel:
    FLAG = None
    inputs = None
    targets = None
    cost = None
    train_op = None
    global_step = None
    input_data = None
    targets_data = None

    def __init__(self, FLAGS):
        self.FLAGS = FLAGS


    def build_model(self, is_training=True, dropout_keep_prob=0.5):
        self.inputs = tf.placeholder(real_type(self.FLAGS), [self.FLAGS.batch_size, 224, 224, 3])
        self.targets = tf.placeholder(tf.int32, [self.FLAGS.batch_size])
    
        with slim.arg_scope(resnet_utils.resnet_arg_scope(is_training)):
            logits, endpoints = resnet_v1.resnet_v1_101(self.inputs, 1000)
        logits = tf.squeeze(logits)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = self.targets)
        self.cost = tf.reduce_sum(loss)
        self.global_step = tf.contrib.framework.get_or_create_global_step()
        self.train_op = tf.train.AdagradOptimizer(0.01).minimize(
            loss, 
            global_step = self.global_step)


    def get_data(self):
        self.inputs_data = np.random.rand(self.FLAGS.batch_size, 224, 224, 3)
        self.targets_data = np.random.randint(1000, size = self.FLAGS.batch_size)

    def get_feed_dict(self):
        return { self.inputs : self.inputs_data, self.targets : self.targets_data }
