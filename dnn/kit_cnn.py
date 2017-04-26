import argparse
import sys

import numpy as np
import tensorflow as tf
import time
import tempfile
from models.cnn import nets_factory
from utils import print_model, real_type
import data_utils.cifar as cifar

FLAGS = None


def main(_):  
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    if FLAGS.job_name == "ps":
        ps_config = tf.ConfigProto(
                gpu_options=tf.GPUOptions(
                    per_process_gpu_memory_fraction=0.01 
                ))

        # Create and start a server for the local task.
        server = tf.train.Server(cluster,
 #                                protocol = "grpc_rdma",
                                 job_name=FLAGS.job_name,
                                 task_index=FLAGS.task_index,
                                 config = ps_config)
        server.join()
    elif FLAGS.job_name == "worker":

        # Create and start a server for the local task.
        server = tf.train.Server(cluster,
#                                 protocol = "grpc_rdma",
                                 job_name=FLAGS.job_name,
                                 task_index=FLAGS.task_index)

        ######################
        # Select the dataset #
        ######################
        # dataset = dataset_factory.get_dataset(FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

        ######################
        # Select the network #
        ######################
        network_fn = nets_factory.get_network_fn(FLAGS.network, FLAGS.num_classes, is_training = True)

        with tf.device(tf.train.replica_device_setter(
                ps_device = '/job:ps/cpu:0',
                worker_device = ("/job:worker/task:%d" % FLAGS.task_index),
                cluster = cluster)):
            
            global_step = tf.contrib.framework.get_or_create_global_step()
            
            images, labels = cifar.distorted_inputs(FLAGS) 
            logits, end_points = network_fn(images)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels)
            cost = tf.reduce_mean(loss)
            train_op = tf.train.AdagradOptimizer(0.01).minimize(cost, global_step = global_step)

        print_model()

        train_dir = tempfile.mkdtemp()

        sess_config = tf.ConfigProto(
            allow_soft_placement=True, 
            log_device_placement=False,
            device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index],

            graph_options=tf.GraphOptions(
                optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L1)
            ),

            gpu_options=tf.GPUOptions(
                visible_device_list=""
            )
        )

        if FLAGS.infer_shapes == True:
            sess_config.graph_options.infer_shapes = FLAGS.infer_shapes
         
        sv = tf.train.Supervisor(
            is_chief = (FLAGS.task_index == 0),
            logdir = train_dir,
            init_op = tf.global_variables_initializer(),
            global_step = global_step,
            summary_writer = None,
            saver = None)

        if FLAGS.task_index == 0:
            print("Worker %d: Initializing session..." % FLAGS.task_index)
        else:
            print("Worker %d: Waiting for session to be initialized..." % FLAGS.task_index)
               
        sess = sv.prepare_or_wait_for_session(server.target, config = sess_config, start_standard_services = True)

        print ("Start warmup %d epoch." % FLAGS.warmup)
        for _ in range(FLAGS.warmup):
            sess.run(train_op)

        current_step = 0
        duration = 0
        while current_step < FLAGS.epoch:
            current_step += 1
            start_time = time.time()
            _, step_loss = sess.run([train_op, cost])
            end_time = time.time()
            print("Finish step %d, loss = %f, speed = %f sampes/s, duration = %f seconds" % (current_step, step_loss, FLAGS.batch_size / (end_time - start_time), end_time - start_time))
            duration += end_time - start_time

        print ("Total Time = %f s." % duration)
        #writer.close()

    else:
        sys.exit("Invalid job role name [%s]!" % args.job_name)

 
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  # Flags for defining the tf.train.ClusterSpec
  parser.add_argument(
      "--ps_hosts",
      type=str,
      default="localhost:8700",
      help="Comma-separated list of hostname:port pairs"
  )

  parser.add_argument(
      "--worker_hosts",
      type=str,
      default="localhost:8800",
      help="Comma-separated list of hostname:port pairs"
  )

  parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
  )


  # Flags for defining the tf.train.Server
  parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
  )


  # Flags for algorithm
  parser.add_argument(
      "--infer_shapes",
      type=bool,
      default=False,
      help="if use rdma"
  )

  parser.add_argument(
      "--network", "-g",
      type=str,
      default="lstm",
      help="lstm/fc/alexnet/vgg19/inception_v3"
  )

  parser.add_argument(
      "--data_dir",
      type=str,
      default="./data",
      help="Training data path"
  )

  parser.add_argument(
      "--use_fp16",
      type=bool,
      default=False,
      help="Train using 16-bit floats instead of 32bit floats"
  )

  parser.add_argument(
      "--epoch", "-e",
      type=int,
      default=3,
      help="number of epoch"
  )

  parser.add_argument(
      "--warmup", "-w",
      type=int,
      default=0,
      help="Warm up epoch"
  )


  parser.add_argument(
      "--batch_size", "-b",
      type=int,
      default=100,
      help="Batch Size"
  )


  parser.add_argument(
      "--num_steps", "-n",
      type=int,
      default=10,
      help="number of step for rnn"
  )

  parser.add_argument(
      "--hidden_size",
      type=int,
      default=200,
      help="Hidden Cell Size for rnn and cnn"
  )

  parser.add_argument(
      "--vocab_size", "-v",
      type=int,
      default=1000,
      help="Vocabulary Size for run"
  )

  parser.add_argument(
      "--num_layers", "-l",
      type=int,
      default=3,
      help="Layers number for rnn and cnn"
  )

  parser.add_argument(
      "--num_classes", "-c",
      type=int,
      default=10,
      help="Class number for cnn."
  )

  parser.add_argument(
      "--num_features", "-f",
      type=int,
      default=784,
      help="Input features for cnn."
  )

### for seq2seq
  parser.add_argument("--learning_rate", type=float, default = 0.5, help = "Learning rate.")
  parser.add_argument("--learning_rate_decay_factor", type=float, default = 0.99, help = "Learning rate decays by this much.")
  parser.add_argument("--max_gradient_norm", type=float, default = 5.0, help = "Clip gradients to this norm.")
  parser.add_argument("--size", type=int, default = 1024, help = "Size of each model layer.")
  parser.add_argument("--from_vocab_size", type=int, default = 40000, help = "English vocabulary size.")
  parser.add_argument("--to_vocab_size", type=int, default = 40000, help = "French vocabulary size.")
  parser.add_argument("--train_dir", type=str, default = "/tmp", help = "Training directory.")
  parser.add_argument("--from_train_data", type=str, default = None,  help = "Training data.")
  parser.add_argument("--to_train_data", type=str, default = None, help = "Training data.")
  parser.add_argument("--from_dev_data", type=str, default = None, help = "Training data.")
  parser.add_argument("--to_dev_data", type=str, default = None, help = "Training data.")
  parser.add_argument("--max_train_data_size", type = int, default = 0, help = "Limit on the size of training data (0: no limit).")
  parser.add_argument("--steps_per_checkpoint", type = int, default = 200, help = "How many training steps to do per checkpoint.")
  parser.add_argument("--decode", type = bool, default = False, help = "Set to True for interactive decoding.")

  FLAGS, unparsed = parser.parse_known_args()
  print("FLAGS = ", FLAGS)

  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
