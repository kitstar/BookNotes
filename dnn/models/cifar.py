import argparse
import sys

import numpy as np
import tensorflow as tf
import utils
import tensorflow.contrib.slim as slim
from utils import real_type


def _variable_on_cpu(FLAGS, name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(FLAGS, name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      FLAGS,
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var
 

class KitModel:
    FLAG = None
    inputs = None
    targets = None
    cost = None
    train_op = None
    global_step = None
    input_data = None
    target_data = None

    logits = None

    def __init__(self, FLAGS):
        self.FLAGS = FLAGS

    def inference(self, images):
        self.input_data = images
        # conv1
        with tf.variable_scope('conv1') as scope:
            kernel = _variable_with_weight_decay(self.FLAGS,
                                                 'weights',
                                                 shape=[5, 5, 3, 64],
                                                 stddev=5e-2,
                                                 wd=0.0)
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(self.FLAGS, 'biases', [64], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation, name=scope.name)

        # pool1
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool1')
        # norm1
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm1')

        # conv2
        with tf.variable_scope('conv2') as scope:
            kernel = _variable_with_weight_decay(self.FLAGS,
                                                 'weights',
                                                 shape=[5, 5, 64, 64],
                                                 stddev=5e-2,
                                                 wd=0.0)
            conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(self.FLAGS, 'biases', [64], tf.constant_initializer(0.1))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(pre_activation, name=scope.name)

        # norm2
        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm2')
        # pool2
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        # local3
        with tf.variable_scope('local3') as scope:
            # Move everything into depth so we can perform a single matrix multiply.
            reshape = tf.reshape(pool2, [self.FLAGS.batch_size, -1])
            dim = reshape.get_shape()[1].value
            weights = _variable_with_weight_decay(self.FLAGS, 'weights', shape=[dim, 384],
                                                  stddev=0.04, wd=0.004)
            biases = _variable_on_cpu(self.FLAGS, 'biases', [384], tf.constant_initializer(0.1))
            local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

        # local4
        with tf.variable_scope('local4') as scope:
            weights = _variable_with_weight_decay(self.FLAGS, 'weights', shape=[384, 192],
                                                  stddev=0.04, wd=0.004)
            biases = _variable_on_cpu(self.FLAGS, 'biases', [192], tf.constant_initializer(0.1))
            local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

        # linear layer(WX + b),
        # We don't apply softmax here because
        # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
        # and performs the softmax internally for efficiency.
        with tf.variable_scope('softmax_linear') as scope:
            weights = _variable_with_weight_decay(self.FLAGS, 'weights', [192, self.FLAGS.num_classes],
                                                  stddev=1/192.0, wd=0.0)
            biases = _variable_on_cpu(self.FLAGS, 'biases', [self.FLAGS.num_classes],
                                      tf.constant_initializer(0.0))
            self.logits  = tf.add(tf.matmul(local4, weights), biases, name=scope.name)

        return self.logits



    def loss(self, labels):
        self.target_data = labels
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.logits, labels = self.target_data)
        self.cost = tf.reduce_mean(loss)
        return self.cost

    def train(self):
        self.global_step = tf.contrib.framework.get_or_create_global_step()
        self.train_op = tf.train.AdagradOptimizer(0.01).minimize(
            self.cost, 
            global_step = self.global_step)
        return self.train_op
