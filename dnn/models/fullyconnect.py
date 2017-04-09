import argparse
import sys

import numpy as np
import tensorflow as tf
import utils
import tensorflow.contrib.slim as slim
from utils import real_type

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

    def build_model(self):
        self.inputs = tf.placeholder(real_type(self.FLAGS), [self.FLAGS.batch_size, self.FLAGS.num_features])
        self.targets = tf.placeholder(tf.int32, [self.FLAGS.batch_size])
    
        with tf.variable_scope("Fully_connected"):
            with slim.arg_scope([slim.fully_connected], 
                                activation_fn = tf.nn.sigmoid, reuse = False):
                net = slim.fully_connected(self.inputs, self.FLAGS.hidden_size, scope = 'input_layer')
                net = slim.repeat(net, self.FLAGS.num_layers - 2, slim.fully_connected, self.FLAGS.hidden_size, scope = 'hidden_layer')
                net = slim.fully_connected(net, self.FLAGS.num_classes, activation_fn = None, scope = 'output_layer')
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = net, labels = self.targets)
                self.cost = tf.reduce_sum(loss)
                self.global_step = tf.contrib.framework.get_or_create_global_step()
                self.train_op = tf.train.AdagradOptimizer(0.01).minimize(
                    loss, 
                    global_step = self.global_step)


    def get_data(self):
        self.inputs_data = np.random.rand(self.FLAGS.batch_size, self.FLAGS.num_features)
        self.targets_data = np.random.rand(self.FLAGS.batch_size)

    def get_feed_dict(self):
        return { self.inputs : self.inputs_data, self.targets : self.targets_data }
