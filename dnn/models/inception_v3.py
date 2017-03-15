import argparse
import sys

import numpy as np
import tensorflow as tf
import utils
import tensorflow.contrib.slim as slim
from utils import real_type
from tensorflow.contrib.slim.python.slim.nets import inception_v3

def vgg_arg_scope(weight_decay=0.0005):
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



def build_model(FLAGS, is_training=True, dropout_keep_prob=0.5):
    inputs = tf.placeholder(real_type(FLAGS), [FLAGS.batch_size, 299, 299, 3])
    targets = tf.placeholder(tf.int32, [FLAGS.batch_size])
    
    logits, endpoints = inception_v3.inception_v3(inputs, FLAGS.num_classes)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets)
    cost = tf.reduce_sum(loss)
    global_step = tf.contrib.framework.get_or_create_global_step()
    train_op = tf.train.AdagradOptimizer(0.01).minimize(loss, global_step=global_step)
 
    return (inputs, targets, train_op)


def get_data(FLAGS):
    inputs_data = np.random.rand(FLAGS.batch_size, 299, 299, 3)
    targets_data = np.random.rand(FLAGS.batch_size)
    return (inputs_data, targets_data)
