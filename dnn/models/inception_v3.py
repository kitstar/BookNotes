import argparse
import sys

import numpy as np
import tensorflow as tf
import utils
import tensorflow.contrib.slim as slim
from utils import real_type
from tensorflow.contrib.slim.python.slim.nets import inception_v3



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
        self.inputs = tf.placeholder(real_type(self.FLAGS), [self.FLAGS.batch_size, 299, 299, 3])
        self.targets = tf.placeholder(tf.int32, [self.FLAGS.batch_size])
    
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits, endpoints = inception_v3.inception_v3(self.inputs, self.FLAGS.num_classes)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = self.targets)
        self.cost = tf.reduce_sum(loss)
        self.global_step = tf.contrib.framework.get_or_create_global_step()
        self.train_op = tf.train.AdagradOptimizer(0.01).minimize(
            loss, 
            global_step = self.global_step)


    def get_data(self):
        self.inputs_data = np.random.rand(self.FLAGS.batch_size, 299, 299, 3)
        self.targets_data = np.random.randint(self.FLAGS.num_classes, size = self.FLAGS.batch_size)

    def get_feed_dict(self):
        return { self.inputs : self.inputs_data, self.targets : self.targets_data }
