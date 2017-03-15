import argparse
import sys

import numpy as np
import tensorflow as tf
import utils
import tensorflow.contrib.slim as slim
from utils import real_type



trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)



def alexnet_v2_arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        biases_initializer=tf.constant_initializer(0.1),
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope([slim.conv2d], padding='SAME'):
            with slim.arg_scope([slim.max_pool2d], padding='VALID') as arg_sc:
                return arg_sc


def build_model(FLAGS, is_training=True, dropout_keep_prob=0.5):
    inputs = tf.placeholder(real_type(FLAGS), [FLAGS.batch_size, 224, 224, 3])
    targets = tf.placeholder(tf.int32, [FLAGS.batch_size])
    
    with tf.variable_scope("alexnet_v2"):
        with slim.arg_scope(alexnet_v2_arg_scope()):
            net = slim.conv2d(inputs, 64, [11, 11], 4, padding = 'VALID', scope = 'conv1')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
            net = slim.conv2d(net, 192, [5, 5], scope='conv2')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
            net = slim.conv2d(net, 384, [3, 3], scope='conv3')
            net = slim.conv2d(net, 384, [3, 3], scope='conv4')
            net = slim.conv2d(net, 256, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool5')

            with slim.arg_scope([slim.conv2d],
                                weights_initializer=trunc_normal(0.005),
                                biases_initializer=tf.constant_initializer(0.1)):
                net = slim.conv2d(net, 4096, [5, 5], padding='VALID', scope='fc6')
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                   scope='dropout6')
                net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout7')
                net = slim.conv2d(net, FLAGS.num_classes, [1, 1], activation_fn=None, normalizer_fn=None,
                                  biases_initializer=tf.constant_initializer(0.0), scope='fc8')

                logits = tf.squeeze(net, [1, 2], name = 'fc8/squeezed')

                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets)
                cost = tf.reduce_sum(loss)
                global_step = tf.contrib.framework.get_or_create_global_step()
                train_op = tf.train.AdagradOptimizer(0.01).minimize(loss, global_step=global_step)
    return (inputs, targets, train_op)


def get_data(FLAGS):
    inputs_data = np.random.rand(FLAGS.batch_size, 224, 224, 3)
    targets_data = np.random.rand(FLAGS.batch_size)
    return (inputs_data, targets_data)

