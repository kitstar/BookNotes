import argparse
import sys

import numpy as np
import tensorflow as tf
from utils import print_model, real_type

FLAGS = None


def main(_):  
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)


    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        # Assigns ops to the local worker by default.
        local_worker_device = "/job:worker/task:%d" % FLAGS.task_index
        with tf.device(tf.train.replica_device_setter(
            worker_device=local_worker_device,
            cluster=cluster)):
            
            if FLAGS.network == 'lstm':
                from models.lstm import build_model, get_data
            elif FLAGS.network == 'fc':
                from models.fullyconnect import build_model, get_data
            elif FLAGS.network == 'alexnet':
                from models.alexnet import build_model, get_data
            elif FLAGS.network == 'resnet':
                print("nothing")
            else:
                sys.exit("Invalid network [%s]" % args.arch)
      
            inputs, targets, train_op = build_model(FLAGS)     
            inputs_data, targets_data = get_data(FLAGS)

        # The StopAtStepHook handles stopping after running given steps.
        hooks=[tf.train.StopAtStepHook(num_steps=FLAGS.epoch)]

        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.
        print_model()

        if FLAGS.network == 'lstm':
            inputs_data = np.random.rand(FLAGS.num_steps, FLAGS.batch_size, FLAGS.hidden_size)
            targets_data = np.random.rand(FLAGS.num_steps, FLAGS.batch_size)
        elif FLAGS.network == 'resnet':
            print("nothing")
        
        current_step = 0;
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(FLAGS.task_index == 0),
                                               #checkpoint_dir="/tmp/train_logs",
                                               save_checkpoint_secs=0,
                                               hooks=hooks) as mon_sess:
      
            while not mon_sess.should_stop():
                current_step += 1
                print("Start step %d" % current_step)
                mon_sess.run(train_op, feed_dict={inputs:inputs_data, targets:targets_data})
                print("Finish step %d" % current_step)

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
      "--network", "-g",
      type=str,
      default="lstm",
      help="lstm"
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
      help="Epoch Size"
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

  
  FLAGS, unparsed = parser.parse_known_args()
  print("FLAGS = ", FLAGS)
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
