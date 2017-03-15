import argparse
import sys

import numpy as np
import tensorflow as tf
import utils
import tensorflow.contrib.slim as slim
from utils import real_type
from tensorflow.contrib.slim.python.slim.nets import inception_v3



def build_model(FLAGS, is_training=True, dropout_keep_prob=0.5):
    inputs = tf.placeholder(real_type(FLAGS), [FLAGS.batch_size, 299, 299, 3])
    targets = tf.placeholder(tf.int32, [FLAGS.batch_size])
    
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
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
