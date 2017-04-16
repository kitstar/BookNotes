import argparse
import sys

import numpy as np
import tensorflow as tf
import utils
from utils import real_type



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


    def build_model(self):
        # Build model...
        #embedding = tf.get_variable("embedding", [vocab_size, size], dtype=data_type())
        self.inputs = tf.placeholder(real_type(self.FLAGS), 
                                     shape = (self.FLAGS.num_steps, self.FLAGS.batch_size, self.FLAGS.hidden_size))
        self.targets = tf.placeholder(tf.int32, shape = (self.FLAGS.num_steps, self.FLAGS.batch_size))
                
#        lstm = tf.nn.rnn_cell.LSTMCell(self.FLAGS.hidden_size)
        lstm = tf.contrib.rnn.GRUCell(self.FLAGS.hidden_size)
        stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm] * self.FLAGS.num_layers)  
        initial_state = state = stacked_lstm.zero_state(self.FLAGS.batch_size, real_type(self.FLAGS))

        outputs = []
        with tf.variable_scope("RNN"):
            for time_step in range(self.FLAGS.num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                output, state = stacked_lstm(self.inputs[time_step, :, :], state)
                outputs.append(output)
     
        output = tf.reshape(tf.concat(axis = 1, values = outputs), [-1, self.FLAGS.hidden_size])
        #output = tf.reshape(tf.concat(1, outputs), [-1, self.FLAGS.hidden_size])
        softmax_w = tf.get_variable("softmax_w", [self.FLAGS.hidden_size, self.FLAGS.vocab_size], dtype=real_type(self.FLAGS))
        softmax_b = tf.get_variable("softmax_b", [self.FLAGS.vocab_size], dtype=real_type(self.FLAGS))
        logits = tf.matmul(output, softmax_w) + softmax_b
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self.targets, [-1])],
            [tf.ones([self.FLAGS.num_steps * self.FLAGS.batch_size] , dtype=real_type(self.FLAGS))])
        self.cost = tf.reduce_sum(loss) / self.FLAGS.batch_size
        final_state = state      

        self.global_step = tf.contrib.framework.get_or_create_global_step()

        self.train_op = tf.train.AdagradOptimizer(0.01).minimize(self.cost, global_step=self.global_step)

    def get_data(self):
        self.input_data = np.random.rand(self.FLAGS.num_steps, self.FLAGS.batch_size, self.FLAGS.hidden_size)
        self.target_data = np.random.rand(self.FLAGS.num_steps, self.FLAGS.batch_size)


    def get_feed_dict(self):
        return { self.inputs : self.input_data, self.targets : self.target_data }


