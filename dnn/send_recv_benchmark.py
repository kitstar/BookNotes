from datetime import datetime
import math
import time


import numpy as np
import argparse
import os
import tempfile
import tensorflow.python.platform
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--dim0', '-m', type=int, default=1048576)

parser.add_argument('--dim1', '-n', type=int, default=256)

parser.add_argument('--batch', '-b', type=int, default=10)

parser.add_argument('--ps_hosts',  default="")

parser.add_argument('--worker_hosts',  default="")

parser.add_argument('--job_name',  default="worker", help="Either 'ps' or 'worker'")

parser.add_argument('--task_index', type=int, default=0)

parser.add_argument('--infer_shape', type=bool, default=False)

args = parser.parse_args()

def PrintParameterCount():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        print(shape)
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    print("Parameter Number:" + str(total_parameters))

def time_tensorflow_run(session, target, num_steps,feed_dict=None, info=None):
    num_burn_in = 1
    for i in range(num_burn_in):
        session.run(target, feed_dict=feed_dict)
    start_time = time.time()
    for i in range(num_steps):
        step_start = time.time();
        y, step = session.run(target, feed_dict=feed_dict)
        print(y)
        step_duration = time.time() - step_start
        print("Worker %d: training step %d done (global step: %d), duration=%f" %
            (args.task_index, i, step, step_duration))
    duration = time.time() - start_time
    if info:
        print ('Used time for %s : %f' %(info, duration / num_steps))
    return duration


ps_host = args.ps_hosts.split(',')
worker_host = args.worker_hosts.split(',')
# Create a cluster from the parameter server and worker hosts.
cluster = tf.train.ClusterSpec({"ps":ps_host, "worker": worker_host})

# start a server for a specific task
server = tf.train.Server(cluster,
                       job_name=args.job_name,
                       task_index=args.task_index)

# config
dim0 = args.dim0
dim1 = args.dim1

worker_device = "/job:worker/task:%d" % (args.task_index)
is_chief = (args.task_index == 0)


if args.job_name == "ps":
    server.join()
elif args.job_name == "worker":
    with tf.device(tf.train.replica_device_setter(
        worker_device=worker_device,
        cluster=cluster)):
        # count the number of updates
        global_step = tf.get_variable('global_step', [], initializer = tf.constant_initializer(0), trainable = False)

        mat = tf.Variable(tf.ones([dim0, dim1]))
        y = tf.reduce_sum(mat)
       
        # Async mode
        init_op = tf.global_variables_initializer()

    sv = None
    train_dir = tempfile.mkdtemp()

    # Create a "supervisor", which oversees the training process.
    sv = tf.train.Supervisor(
        is_chief=is_chief,
        logdir=train_dir,
        init_op=init_op, 
        save_model_secs=0,
        recovery_wait_secs=1,
        global_step=global_step)

    # The supervisor takes care of session initialization, restoring from
    # a checkpoint, and closing when done or an error occurs.
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False,
                    graph_options=tf.GraphOptions(
                        optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L1), 
                        infer_shapes=args.infer_shape))

    if is_chief:
      print("Worker %d: Initializing session..." % args.task_index)
    else:
      print("Worker %d: Waiting for session to be initialized..." %args.task_index)

    sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)

    print("Worker %d: Session initialization complete." % args.task_index)

    duration = time_tensorflow_run(sess, [y, global_step], args.batch, {}, '[copy + forward + backward + update]')

    print('********************** Two node Benchmark  **********************')
    print('Avg elasped time per mini-batch (sec/mini-batch): '+str(round(duration / args.batch, 6)) )
