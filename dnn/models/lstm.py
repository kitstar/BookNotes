import argparse
import sys

import numpy as np
import tensorflow as tf
import utils
from utils import real_type
#import utils

def build_model(FLAGS):
    # Build model...
    #embedding = tf.get_variable("embedding", [vocab_size, size], dtype=data_type())
      
    lstm = tf.nn.rnn_cell.LSTMCell(FLAGS.hidden_size)
    stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm] * FLAGS.num_layers)  
    initial_state = state = stacked_lstm.zero_state(FLAGS.batch_size, real_type(FLAGS))

    inputs = tf.placeholder(real_type(FLAGS), [FLAGS.num_steps, FLAGS.batch_size, FLAGS.hidden_size])
    targets = tf.placeholder(tf.int32, [FLAGS.num_steps, FLAGS.batch_size])

    outputs = []
    with tf.variable_scope("RNN"):
        for time_step in range(FLAGS.num_steps):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            output, state = stacked_lstm(inputs[time_step, :, :], state)
            outputs.append(output)
     
    output = tf.reshape(tf.concat(1, outputs), [-1, FLAGS.hidden_size])
    softmax_w = tf.get_variable("softmax_w", [FLAGS.hidden_size, FLAGS.vocab_size], dtype=real_type(FLAGS))
    softmax_b = tf.get_variable("softmax_b", [FLAGS.vocab_size], dtype=real_type(FLAGS))
    logits = tf.matmul(output, softmax_w) + softmax_b
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(targets, [-1])],
        [tf.ones([FLAGS.num_steps * FLAGS.batch_size] , dtype=real_type(FLAGS))])
    cost = tf.reduce_sum(loss) / FLAGS.batch_size
    final_state = state      

    global_step = tf.contrib.framework.get_or_create_global_step()

    train_op = tf.train.AdagradOptimizer(0.01).minimize(loss, global_step=global_step)

    return (inputs, targets, train_op)
