import argparse
import sys

import numpy as np
import tensorflow as tf
import utils
from utils import real_type



def sigmoidDNNLayer(layerIdx, input, inputDim, outputDim, real_type = tf.float32):
    W = tf.get_variable("W" + str(layerIdx), [inputDim, outputDim], dtype = real_type)
    B = tf.get_variable("B" + str(layerIdx), [outputDim], dtype = real_type)
    return tf.nn.sigmoid(tf.nn.xw_plus_b(input, W, B))


def build_model(FLAGS):
    inputs = tf.placeholder(real_type(FLAGS), [FLAGS.batch_size, FLAGS.num_features])
    targets = tf.placeholder(tf.int32, [FLAGS.batch_size])
    layers = [sigmoidDNNLayer(0, inputs, FLAGS.num_features, FLAGS.hidden_size, real_type = real_type(FLAGS))]
    for layer in range(1, FLAGS.num_layers - 1):
        layers.append(sigmoidDNNLayer(layer, layers[layer - 1], FLAGS.hidden_size, FLAGS.hidden_size, real_type = real_type(FLAGS)))

    ow = tf.get_variable("W" + str(FLAGS.num_layers), [FLAGS.hidden_size, FLAGS.num_classes], dtype = real_type(FLAGS))
    ob = tf.get_variable("B" + str(FLAGS.num_layers), [FLAGS.num_classes], dtype = real_type(FLAGS))
    last_layer = tf.nn.xw_plus_b(layers[FLAGS.num_layers - 2], ow, ob)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(last_layer, targets)
    cost = tf.reduce_sum(loss)
    global_step = tf.contrib.framework.get_or_create_global_step()
    train_op = tf.train.AdagradOptimizer(0.01).minimize(loss, global_step=global_step)
    return (inputs, targets, train_op)


def get_data(FLAGS):
    inputs_data = np.random.rand(FLAGS.batch_size, FLAGS.num_features)
    targets_data = np.random.rand(FLAGS.batch_size)
    return (inputs_data, targets_data)

