import argparse
import sys

import numpy as np
import tensorflow as tf
import utils
import tensorflow.contrib.slim as slim
from utils import real_type
from tensorflow.contrib.slim.python.slim.nets import vgg



def vgg_arg_scope(weight_decay = 0.0005):
    """Defines the VGG arg scope.
    Args:
        weight_decay: The l2 regularization coefficient.
    Returns:
        An arg_scope.
    """
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=tf.constant_initializer(0.0)):
        with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
            return arg_sc


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
    
        with slim.arg_scope(vgg_arg_scope()):
            logits, endpoints = vgg.vgg_16(self.inputs, num_classes = self.FLAGS.num_classes)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = self.targets)
        self.cost = tf.reduce_sum(loss)
        self.global_step = tf.contrib.framework.get_or_create_global_step()
        self.train_op = tf.train.AdagradOptimizer(0.01).minimize(
            self.cost, 
            global_step = self.global_step)


    def get_data(self):
        self.inputs_data = np.random.rand(self.FLAGS.batch_size, 224, 224, 3)
        self.targets_data = np.random.randint(self.FLAGS.num_classes, size = self.FLAGS.batch_size)

    def get_feed_dict(self):
        return { self.inputs : self.inputs_data, self.targets : self.targets_data }
