import argparse
import sys

import numpy as np
import tensorflow as tf
import utils
import tensorflow.contrib.slim as slim
from utils import real_type



def build_model(FLAGS):
    inputs = tf.placeholder(real_type(FLAGS), [FLAGS.batch_size, FLAGS.num_features])
    targets = tf.placeholder(tf.int32, [FLAGS.batch_size])
    
    with tf.variable_scope("Fully_connected"):
        with slim.arg_scope([slim.fully_connected], 
                            activation_fn = tf.nn.sigmoid, reuse = False):
            net = slim.fully_connected(inputs, FLAGS.hidden_size, scope = 'input_layer')
            net = slim.repeat(net, FLAGS.num_layers - 2, slim.fully_connected, FLAGS.hidden_size, scope = 'hidden_layer')
            net = slim.fully_connected(net, FLAGS.num_classes, activation_fn = None, scope = 'output_layer')
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(net, targets)
            cost = tf.reduce_sum(loss)
            global_step = tf.contrib.framework.get_or_create_global_step()
            train_op = tf.train.AdagradOptimizer(0.01).minimize(loss, global_step=global_step)
    return (inputs, targets, train_op)


def get_data(FLAGS):
    inputs_data = np.random.rand(FLAGS.batch_size, FLAGS.num_features)
    targets_data = np.random.rand(FLAGS.batch_size)
    return (inputs_data, targets_data)

