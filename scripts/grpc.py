from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dim', type=int, default=1024 * 1024)
parser.add_argument('--dim2', type=int, default=128)
parser.add_argument('--num_steps', type=int, default=100)
parser.add_argument('--job_name', default="worker")
parser.add_argument('--ps_hosts',  default="localhost:3200")
parser.add_argument('--worker_hosts',  default="localhost:3300")
parser.add_argument('--task_index', type=int, default=0)

args = parser.parse_args()

print(args)


#Create a cluster from the parameter server and worker hosts.
ps_hosts = args.ps_hosts.split(",")
worker_hosts = args.worker_hosts.split(",")
cluster = tf.train.ClusterSpec({"ps":ps_hosts, "worker": worker_hosts})
server = tf.train.Server(cluster, job_name=args.job_name, task_index=args.task_index)

if args.job_name == 'ps':
    server.join()
    exit

dim = args.dim
dim2 = args.dim2
num_steps = args.num_steps

print("dim = " + str(dim))
print("dim2 = " + str(dim2))
print("num_steps = " + str(num_steps))

print("Program Start!")
prog_start_time = time.time()

#a = tf.constant(3)
#b = tf.constant(2)
a = tf.Variable(tf.ones([dim, dim2]))

with tf.device("/job:worker/task:0"):
#    x = tf.mul(a, b)
    y = tf.reduce_sum(a)

#with tf.device("/job:worker/task:1"):
#    y = tf.mul(a, b)

with tf.Session(server.target) as sess:
    start_time = time.time()
    iter_start_time = start_time

    sess.run(tf.initialize_all_variables()) 
    for i in range(num_steps):
	print(sess.run([y]))

        duration = time.time() - iter_start_time
        print ('** Used time for %s : %f' %(i, duration))
        iter_start_time = time.time()

    duration = time.time() - start_time
    print ('***** Step avg. time: %f ******' %(duration / num_steps))

print("Finished!")
prog_duration = time.time() - prog_start_time
print('***** Total program time: %f ******' %(prog_duration))
