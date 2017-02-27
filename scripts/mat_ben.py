# highlighted code is MPI specific

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import os
import sys
import argparse

#from tensorflow.python import pywrap_tensorflow as py_tf

print(tf.__version__)
print(tf.__file__)

cluster = tf.train.ClusterSpec({"worker": ["10.0.0.10:9999"], "master": ["10.0.0.9:9998"]})

#print("GetHostRank: " + str(py_tf.TF_GetHostRank()))
#node_index = py_tf.TF_GetHostRank();

parser = argparse.ArgumentParser()
parser.add_argument('--dim', type=int, default=1024 * 1024)
parser.add_argument('--dim2', type=int, default=256)
parser.add_argument('--num_steps', type=int, default=10)
parser.add_argument('--index', type=int, default=0)
args = parser.parse_args()
print(args)

dim = args.dim
dim2 = args.dim2
num_steps = args.num_steps
node_index = args.index

print("dim = " + str(dim))
print("dim2 = " + str(dim2))
print("num_steps = " + str(num_steps))

print("Program Start!")
prog_start_time = time.time()

if(node_index == 0):
    a = tf.Variable(tf.ones([dim, dim2]))

    with tf.device("/job:worker/task:0"):
        y = tf.reduce_sum(a)

    server = tf.train.Server(cluster, job_name="master", protocol = "grpc", task_index=0)

    sess_config = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False,
                    graph_options=tf.GraphOptions(
                        optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L1), 
                        infer_shapes=True))


    with tf.Session(server.target, config=sess_config) as sess:
        start_time = time.time()
        iter_start_time = start_time

        sess.run(tf.initialize_all_variables()) 
        for i in range(num_steps):
            print(sess.run([y]))

            duration = time.time() - iter_start_time
            print ('** Used time for %s : %f' %(i, duration))
            iter_start_time = time.time()

        duration = time.time() - start_time
        print ('***** Step avg.time: %f ******' %(duration / num_steps))

else:
    server = tf.train.Server(cluster, job_name="worker", protocol = "grpc", task_index=node_index - 1)
    server.join()

print("Program Finished!")
prog_duration = time.time() - prog_start_time
print('***** Total program time: %f ******' %(prog_duration))
#sys.exit(0)
os.system('kill %d' % os.getpid())
