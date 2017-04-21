import multiprocessing
import os
import sys
import tensorflow as tf
import threading
import time
from datetime import datetime
from tensorflow.python.client import timeline

from lib import AdsDataReader, AdsDnnModel
from util import clean_output, latest_checkpoint, print_model

from tensorflow.python.platform import gfile

# IO arguments
tf.app.flags.DEFINE_string("data_dir", "data", "File path of input data.")
tf.app.flags.DEFINE_string("data_name", "dnn_small_data_30k_tf_record", "File path of input data.")
tf.app.flags.DEFINE_string("log_dir", ".", "Directory path storing log files.")
tf.app.flags.DEFINE_string("model_dir", "model", "Directory path storing model files.")
tf.app.flags.DEFINE_string("output_dir", "output", "Directory path of evalutation output.")
tf.app.flags.DEFINE_string("output_name", "prev.tsv", "File name of evalutation output.")

# Model arguments
tf.app.flags.DEFINE_string("model_type", "AdsDnn", "one of AdsDnn ResNet")
tf.app.flags.DEFINE_boolean("is_validation", False, "Validation or train.")
tf.app.flags.DEFINE_boolean("use_reduce_sum", True, "Validation or train.")
tf.app.flags.DEFINE_integer("batch_size", 192, "Count of examples fed to calculate gradients.")
tf.app.flags.DEFINE_string("feature_group_dimension", "36,880,145250,263017,2740470,3280567,13901203,17276811,25085303,25085825,25087055,28318928,28927038,29662459,30668362,32792210,39094534,40140145,40863511,41277438", "")
tf.app.flags.DEFINE_integer("embedding_count", 5, "Dimension count of group embedding")
tf.app.flags.DEFINE_integer("layer1_count", 300, "Dimension count of layer1")
tf.app.flags.DEFINE_integer("layer2_count", 150, "Dimension count of layer2")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "Leaning rate for optimizers.")
tf.app.flags.DEFINE_integer("iteration", 1, "num of epochs before stopping")
tf.app.flags.DEFINE_integer("save_model_secs", 0, "Time interval of saving model into checkpoint and the unit is second.")

# Distributed arguments
tf.app.flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs.")
tf.app.flags.DEFINE_integer("ps_port", 2200, "parameter port for each host")
tf.app.flags.DEFINE_string("worker_hosts", "localhost", "Comma-separated list of hostname:port pairs.")
tf.app.flags.DEFINE_integer("worker_per_host", 2, "worker count per host")
tf.app.flags.DEFINE_integer("worker_port", 2300, "worker start port each host, if two workers per host, then [worker_port, worker_port + 1]")
tf.app.flags.DEFINE_string("job_name", "ps", "One of 'ps', 'worker'.")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job.")
tf.app.flags.DEFINE_integer("ps_intra_parallel", 0, "")
tf.app.flags.DEFINE_integer("ps_inter_parallel", 0, "")
tf.app.flags.DEFINE_integer("wk_intra_parallel", 1, "")
tf.app.flags.DEFINE_integer("wk_inter_parallel", 1, "")
tf.app.flags.DEFINE_boolean("use_bfloat16_transfer", False, "Use bfloat16 to reduce network traffic")
tf.app.flags.DEFINE_string("optimizer", "AdaGrad", "Optimizer type, options are: AdaGrad, FTRL, RMSProp, GradientDecent")
tf.app.flags.DEFINE_float("success_ratio", 0.95, "master work will exit after success_ratio workers finished")
tf.app.flags.DEFINE_integer("session_run_timeout", 2000, "session single step timeout in ms")
tf.app.flags.DEFINE_integer("wait_for_exit_timeout", 1800, "wait seconds before force exiting")
tf.app.flags.DEFINE_integer("stop_after_timeout_count", 100, "stop training after N timeout count")
tf.app.flags.DEFINE_boolean("shard_saving", False, "shard checkpoint saving")
tf.app.flags.DEFINE_boolean("checkpoint_all_global_variables", False, "checkpoint all global variables including training variables")
tf.app.flags.DEFINE_boolean("max_checkpoint", 1, "Keep max checkpoint count")
tf.app.flags.DEFINE_integer("ps_per_host", 1, "parameter server count per host")
tf.app.flags.DEFINE_float("initial_weight_range", 0.05, "weight initial weight range")
tf.app.flags.DEFINE_float("exit_after_per_host_failure", 0.5, "exit after per host failure")
tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "allow soft placement")
tf.app.flags.DEFINE_boolean("enable_jit", False, "enable XLA jit")
# Debug arguments
tf.app.flags.DEFINE_integer("report_progress_step", 100, "")
tf.app.flags.DEFINE_boolean("log_device_placement", False, "Print placement of variables and operations.")
tf.app.flags.DEFINE_integer("log_timeline_step", 1000, "log time each N steps")

tf.app.flags.DEFINE_integer("max_step", 0, "max step to stop")

# Additional arguments for validation
tf.app.flags.DEFINE_integer("ps_count", 0, "ps count for training and validation")

tf.app.flags.DEFINE_boolean("infer_shapes", False, "If enable rdma.")

FLAGS = tf.app.flags.FLAGS
LABEL_COUNT = 1

def get_non_checkpointable_variables():
    checkpoint_backlist = ["finish_count", "exit_signal"]
    return list(filter(lambda x: x.name in checkpoint_backlist, tf.global_variables()))

def get_non_trainable_variables():
    trainable_variables = tf.trainable_variables()
    return list(filter(lambda x: x not in trainable_variables, tf.global_variables()))

def get_checkpointable_variables():
    checkpoint_backlist = ["finish_count", "exit_signal"]
    return list(filter(lambda x: x.op.name not in checkpoint_backlist, tf.global_variables()))

def create_graph(filename_queue, ps_count):
    '''
    Isolate the codes of creating graph for consistency so that evaluation code can restore the checkpoint file from training code.
    '''
    #reader = AdsDataReader(filename_queue)
    print(datetime.now(), "feature group dimension:", FLAGS.feature_group_dimension)
    model = AdsDnnModel(filename_queue=filename_queue,
                       feature_dimensions=[int(x) for x in FLAGS.feature_group_dimension.strip().split(",")],
                       batch_size=FLAGS.batch_size,
                       embedding_count=FLAGS.embedding_count,
                       layer1_count=FLAGS.layer1_count,
                       layer2_count=FLAGS.layer2_count,
                       label_count=LABEL_COUNT,
                       learning_rate=FLAGS.learning_rate,
                       ps_count=ps_count,
                       use_reduce_sum=FLAGS.use_reduce_sum,
                       optimizer=FLAGS.optimizer)

    saver = tf.train.Saver(get_checkpointable_variables() if FLAGS.checkpoint_all_global_variables else tf.trainable_variables(),
                            reshape=True,
                            allow_empty=True,
                            max_to_keep=FLAGS.max_checkpoint,
                            sharded=FLAGS.shard_saving)
    return model, saver

def get_app_id():
    app_id = os.environ.get("APP_ID")
    if not app_id:
        app_id = "None"
    return app_id

def evalute():
    # Prepare graph and configuration
    data_files = list(filter(lambda x: not gfile.IsDirectory(x), [os.path.join(FLAGS.data_dir, x) for x in gfile.ListDirectory(FLAGS.data_dir)]))
    filename_queue = tf.train.string_input_producer(data_files, num_epochs=1)
    model, saver = create_graph(filename_queue, FLAGS.ps_count)
    init_op = tf.global_variables_initializer()
    local_variables_init_op = tf.local_variables_initializer()
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=FLAGS.log_device_placement)
    output_path = os.path.join(FLAGS.output_dir, FLAGS.output_name)

    with tf.Session(config=sess_config) as sess, open(output_path, "w") as filew:
        sess.run([init_op, local_variables_init_op])

        # Restore checkpoint
        checkpoint_path = latest_checkpoint(FLAGS.model_dir)
        if checkpoint_path:
            print("Restoring model from", checkpoint_path)
            try:
                saver.restore(sess, checkpoint_path)
            except tf.errors.NotFoundError as e:
                print(datetime.now(), "Not all variables found in checkpoint, ingore") 
            print_model(sess)
        else:
            raise IOError("No model files found")

        # Populate the queues
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Start to validate
        local_step = 0
        graph_comptime = 0.0
        evaluation_start = time.time()
        try:
            while not coord.should_stop():
                # Exectuion OPs
                begin = time.time()
                # _loss, _pred, _labels, _weights, final, term_embeddings, layer1_output, layer3_input1, layer3_input2, pos, layer2_output, layer2_input, embedding_out = sess.run(
                #     [
                #         model.loss_fn,
                #         model.prediction,
                #         model.labels,
                #         model.weights,
                #         model.final,
                #         model.term_embeddings,
                #         model.layer1_output,
                #         model.layer3_input1,
                #         model.layer3_input2,
                #         model.pos,
                #         model.layer2_output,
                #         model.layer2_input,
                #         model.embedding_out,
                #     ])
                _pred, _labels, _weights = sess.run([model.prediction, model.labels, model.weights])
                end = time.time()

                # Update counters
                local_step += 1
                graph_comptime += end - begin
                if local_step % FLAGS.report_progress_step == 0:
                    print("local_step: %d, mean_step_time: %.3f, validation_time: %.2f" % (local_step, graph_comptime/local_step, end-evaluation_start))
                    #print("Pred:", _pred, "Loss:", _loss)
                    #print("layer3_input1:", layer3_input1)
                    #print("embedding_out:", embedding_out)
                    #print("term embedding:", term_embeddings)
                    #print("layer2_input:", layer2_input)
                    #print("layer1_output:", layer1_output)
                    #print("layer2_output:", layer2_output)
                    #print("layer3_input2:", layer3_input2)
                    #print("final:", final)
                    #print("pos:", pos)
                    #print(_loss)
                    #print(_pred)

                # Output result
                for i in range(len(_pred)):
                    filew.write(str(_pred[i][0]) + '\t' + str(_labels[i][0]) + '\t' + str(_weights[i][0]) + '\n')
        except tf.errors.OutOfRangeError:
            print("Reach EOF of data")

        # Stop all threads
        coord.request_stop()
        coord.join(threads)

def train_dist(process_index):
    # redirect stdout stderr to files
    task_index = FLAGS.task_index * FLAGS.worker_per_host + process_index
    log_file_name = os.path.join(FLAGS.log_dir, "worker_%d.log" % task_index)
    sys.stdout = open(log_file_name, 'w', 1)
    sys.stderr = sys.stdout

    ps_servers, worker_servers = get_cluster_servers()

    print(datetime.now(), "Identify role, host worker:%d started, task_index:%d, Pid:%d" % (process_index, task_index, os.getpid()))
    cluster = tf.train.ClusterSpec({"ps": ps_servers, "worker": {task_index: worker_servers[task_index]}})
    is_chief = task_index == 0

    ps_count = len(cluster.job_tasks("ps"))

    # Start GRPC server
    print(datetime.now(), "Start GRPC server")
    if FLAGS.infer_shapes == True:
    	sess_config.graph_options.infer_shapes = FLAGS.infer_shapes
    sess_config = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                 log_device_placement=FLAGS.log_device_placement,
                                 intra_op_parallelism_threads=FLAGS.wk_intra_parallel,
                                 inter_op_parallelism_threads=FLAGS.wk_inter_parallel,
                                 graph_options=tf.GraphOptions(enable_bfloat16_sendrecv=FLAGS.use_bfloat16_transfer))
    if FLAGS.enable_jit:
        sess_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    server = tf.train.Server(cluster, job_name="worker", task_index=task_index, config = sess_config)

    worker_device = "/job:worker/task:%d" % task_index
    data_file_name = os.path.join(FLAGS.data_dir, "part-{0}".format(task_index))
    if not gfile.Exists(data_file_name):
        data_file_name = os.path.join(FLAGS.data_dir, "part-{0:0>5}".format(task_index))
    
    with tf.device(worker_device):
        filename_queue = tf.train.string_input_producer([data_file_name], num_epochs=FLAGS.iteration)

    # Create graph
    print("%s: Create graph" % str(datetime.now()))
    device_setter = tf.train.replica_device_setter(worker_device=worker_device, ps_device='/job:ps/cpu:0', cluster=cluster)
    with tf.device(device_setter):
        finish_count = tf.Variable(0, name="finish_count", trainable=False)
        exit_signal = tf.Variable(0, name="exit_signal", trainable=False)
        worker_finish_op = finish_count.assign_add(1, use_locking=False)
        exit_signal_op = exit_signal.assign_add(1, use_locking=False)
        finish_count_reset = finish_count.assign(0)
        exit_signal_reset = exit_signal.assign(0)
        #if ps_count != FLAGS.ps_count:
        #    print("%s: FLAGS.ps_count != ps_count" % str(datetime.now()))
        #    sys.exit(-1)
#        with tf.device('/cpu:0'):
        model, saver = create_graph(filename_queue, ps_count)

        ## initialize all variables
        init_global_variables_op = tf.global_variables_initializer()
    
    report_uninitialized_op = tf.report_uninitialized_variables(tf.global_variables())

    initializer_dict = {x.op.name : x.initializer for x in tf.global_variables()}

    print(initializer_dict.keys())
    def restore_or_initialize_variables(sess):
        print("calling restore and initialize variables")
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt:
            print(datetime.now(), "checkpoint found, restoring, {0} {1}".format(ckpt.model_checkpoint_path, ckpt.all_model_checkpoint_paths))
            try:
                saver.restore(sess, ckpt.model_checkpoint_path)
                #saver.recover_last_checkpoints(ckpt.all_model_checkpoint_paths)
            except tf.errors.NotFoundError as e:
                print(datetime.now(), "Not all variables found in checkpoint, ingore")

            # get all uninitialized variables and initialize them
            uninitialized_tensor = sess.run(report_uninitialized_op)
            print("uninitialized tensor:", uninitialized_tensor)
            uninitialized_variables_list = uninitialized_tensor.tolist()
            for v in uninitialized_variables_list:
                variable_name = v.decode("utf-8")
                print(datetime.now(), "variable:%s not initialized, initialize it" % variable_name)
                sess.run(initializer_dict[variable_name])
        else:
            print(datetime.now(), "no checkpoint found, initialize all global variables")
            sess.run(init_global_variables_op)

        print_model(sess)

    sv = tf.train.Supervisor(is_chief=is_chief,
                             init_op=None,
                             init_fn=restore_or_initialize_variables,
                             local_init_op=tf.local_variables_initializer(),
                             logdir=FLAGS.model_dir,
                             global_step=model.global_step,
                             saver=None, # Disable the saver of Supervisor which separates checkpoint files cross machines!
                             save_model_secs=None, # if FLAGS.save_model_secs<=0 else FLAGS.save_model_secs,
                             summary_writer=None)
    with sv.prepare_or_wait_for_session(master=server.target, config=sess_config, start_standard_services=True) as sess:
        if is_chief:
            sess.run([finish_count_reset, exit_signal_reset])
        print(datetime.now(), "Create session success")
        local_step = 0
        graph_comptime = 0.0
        #step_count = sess.run(model.global_step)
        train_start = time.time()
        local_host_loss = 0
        report_progress_sum_loss = 0
        last_global_step = sess.run(model.global_step)
        last_timestamp = time.time()

        sess_timeout_count = 0
        is_eof = False

        try:
            while not sv.should_stop() and sess_timeout_count < FLAGS.stop_after_timeout_count:
                if FLAGS.max_step != 0 and local_step > FLAGS.max_step:
                    print("max_step is defined and stop after", FLAGS.max_step)
                    break
                # Exectuion OPs
                begin = time.time()
                #_, step_loss, step_count, final, _pos, _pos_tensor, _sample_weights, _labels, _unweighted_loss, _final_loss = sess.run([
                #    model.train_op, model.loss_fn, model.global_step, model.final, model.pos, model.pos_tensor,
                #    model.weights, model.labels, model.unweighted_loss, model.final_loss,
                #    ])
                #run_metadata = tf.RunMetadata()
                #term_shapes, step_count = sess.run([model.term_shapes, model.global_step], options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, timeout_in_ms=300), run_metadata=run_metadata)
                #step_loss = 100
                #_, step_loss, step_count = sess.run([model.train_op, model.loss_fn, model.global_step], options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_metadata)
                try:
                    _, step_loss, step_count = sess.run([model.train_op, model.loss_fn, model.global_step], options=tf.RunOptions(timeout_in_ms=FLAGS.session_run_timeout)) #, run_metadata=run_metadata)
                except tf.errors.DeadlineExceededError:
                    print(datetime.now(), "session run deadline exceed:", FLAGS.session_run_timeout, "session timeout count:", sess_timeout_count)
                    sess_timeout_count += 1
                    continue
                end = time.time()
                local_host_loss += step_loss
                report_progress_sum_loss += step_loss

                # Update counters
                local_step += 1
                graph_comptime += end - begin
                train_time = end - train_start
                if local_step % FLAGS.report_progress_step == 0:
                    print("%s: global_step: %d, local_step:%d, mean_step_loss: %.3f, local_step_loss: %.3f, mean_step_time: %.3f, train_time:%.2f, QPS:%.2f" % (str(datetime.now()), step_count, local_step, local_host_loss / local_step, report_progress_sum_loss / FLAGS.report_progress_step, graph_comptime/local_step, train_time, (step_count - last_global_step) * FLAGS.batch_size / (end - last_timestamp)))
                    last_global_step = step_count
                    last_timestamp = end
                    report_progress_sum_loss = 0
                    #if local_step % FLAGS.log_timeline_step == 0:
                    #    trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                    #    trace_file = open(os.path.join(FLAGS.log_dir, "timeline-%d-%d.ctf.json" % (task_index, local_step)), 'w')
                    #    trace_file.write(trace.generate_chrome_trace_format())
                    #    trace_file.close()
                    #print(type(_unweighted_loss), dir(_unweighted_loss))
                    #print("unweighted loss shape:", _unweighted_loss.shape, "final loss shape:", _final_loss.shape, "sample weight shape:", _sample_weights.shape)
                    #print(_unweighted_loss)
                    #print(_final_loss)
                    #print(_labels)
                    #print(_pos)
                    #print(_pos_tensor)
                    #print("final layer parameter:", final)
                
                # write checkpoint
                #saver.save(sess, os.path.join(FLAGS.model_dir, "model.ckpt"), model.global_step)
                #print(datetime.now(), "saving checkpoint done")
                #print("lasted checkpoint file:", tf.train.latest_checkpoint(FLAGS.model_dir))
                #print("last checkpoints:", saver.last_checkpoints)
                #checkpoint_count += 1
        except tf.errors.OutOfRangeError:
            print("%s: Reach EOF of data" % str(datetime.now()))
            is_eof = True

        # Exit process
        print("%s: Worker-%d finish training, start to wait for terminate signal" % (str(datetime.now()), task_index))
        worker_count = len(worker_servers)
        exit_after_worker_count = FLAGS.success_ratio * worker_count
        print("%s: Cluster worker count: %d, exit after %d workers finished" % (str(datetime.now()), worker_count, exit_after_worker_count))
        if sess_timeout_count >= FLAGS.stop_after_timeout_count:
            print(datetime.now(), "Training failed! Stop after max session timeout count:", FLAGS.stop_after_timeout_count)
        else:
            print(datetime.now(), "Training success! Increase finish_count")
            sess.run(worker_finish_op)
        
        if is_chief:
            print("%s: Is chief worker, wait for all other workers to finish training" % str(datetime.now()))
            # After all workers finish, worker0 will save the model
            wait_for_exit_start = time.time()
            while True:
                returned_finish_count = sess.run(finish_count)
                if returned_finish_count < exit_after_worker_count:
                    print(datetime.now(), "Master worker. Success worker count: %d < exit_after_worker_count:%d" % (returned_finish_count, exit_after_worker_count))
                    time.sleep(30)
                else:
                    break
                if time.time() - wait_for_exit_start > FLAGS.wait_for_exit_timeout:
                    print(datetime.now(), "Already waited %d seconds, force exit!" % FLAGS.wait_for_exit_timeout)
                    sys.exit("Force exit")
            print(datetime.now(), "%d workers done. Save model" % returned_finish_count)
            saver.save(sess, os.path.join(FLAGS.model_dir, "model.ckpt"), model.global_step) # Call saver explicitly will output checkpint files in current machine!
            sess.run(exit_signal_op) # Tell other workers to exit
        else:
            while True:
                returned_exit_signal = sess.run(exit_signal)
                if returned_exit_signal == 0:
                    print(datetime.now(), "NonMaster worker, no exit signal, success worker count: %d" % sess.run(finish_count))
                    time.sleep(30)
                else:
                    print(datetime.now(), "NonMaster worker, exit signal, exit! Success worker count: %d" % sess.run(finish_count))
                    break
        sv.stop()

def monitor_children(ps_processes, worker_processes):
    print(datetime.now(), "monitoring all children processes")

    begin = time.time()
    alive_ps = {}
    alive_worker = {}
    [alive_ps.update({i : ps_processes[i]}) for i in range(len(ps_processes))]
    [alive_worker.update({i : worker_processes[i]}) for i in range(len(worker_processes))]
    
    worker_failure_count = 0

    while True:
        time.sleep(15)
        alive_ps_ids = list(alive_ps.keys())
        for id in alive_ps_ids:
            if not alive_ps[id].is_alive():
                exit_code = alive_ps[id].exitcode
                if exit_code != 0:
                    sys.exit("Exit due to ps failure")

                print(datetime.now(), "ps inner_index:%d Pid:%d exit with code:%d, runtime:%3.f seconds" % (id, alive_ps[id].pid, exit_code, time.time() - begin))
                alive_ps.pop(id)
        
        alive_worker_ids = list(alive_worker.keys())
        for id in alive_worker_ids:
            if not alive_worker[id].is_alive():
                exit_code = alive_worker[id].exitcode
                print(datetime.now(), "worker inner_index:%d Pid:%d exit with code:%d, runtime:%3.f seconds" % (id, alive_worker[id].pid, exit_code, time.time() - begin))
                alive_worker.pop(id)
                
                if worker_failure_count != 0:
                    worker_failure_count += 1
                    if len(worker_processes) * FLAGS.exit_after_per_host_failure <= worker_failure_count:
                        sys.exit("Exit due to too many failures")
                if id == 0 and FLAGS.task_index == 0:
                    print(datetime.now(), "master worker exit, exit process, kill all alive children")
                    [p.terminate() for p in ps_processes]
                    [p.terminate() for p in worker_processes]
                    if exit_code != 0:
                        print(datetime.now(), "master worker failed")
                        sys.exit("master worker failed")
                    return

def get_machine_list():
    ps_hosts = FLAGS.ps_hosts
    worker_hosts = FLAGS.worker_hosts
    if ps_hosts == "":
        ps_hosts = worker_hosts
    ps_ips = ps_hosts.strip().split(",")
    worker_ips = worker_hosts.strip().split(",")
    return ps_ips, worker_ips
    
def get_cluster_servers():
    ps_hosts = FLAGS.ps_hosts
    worker_hosts = FLAGS.worker_hosts
    if ps_hosts == "":
        ps_hosts = worker_hosts
    ps_ips = ps_hosts.strip().split(",")
    worker_ips = worker_hosts.strip().split(",")

    ps_servers = []
    for ip in ps_ips:
        ps_servers.extend([ip + ":" + str(FLAGS.ps_port + i) for i in range(FLAGS.ps_per_host)])

    worker_servers = []
    for ip in worker_ips:
        worker_servers.extend([ip + ":" + str(FLAGS.worker_port + i) for i in range(FLAGS.worker_per_host)])

    return ps_servers, worker_servers

def ps(inner_host_index):
    ps_servers, worker_servers = get_cluster_servers()

    cluster = tf.train.ClusterSpec({"ps": ps_servers, "worker": worker_servers})
    ps_index = FLAGS.task_index * FLAGS.ps_per_host + inner_host_index
    
    log_file_name = os.path.join(FLAGS.log_dir, "ps_%d.log" % ps_index)
    sys.stdout = open(log_file_name, 'w', 1)
    sys.stderr = sys.stdout

    server_config = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                   log_device_placement=FLAGS.log_device_placement,
                                   intra_op_parallelism_threads=FLAGS.ps_intra_parallel,
                                   inter_op_parallelism_threads=FLAGS.ps_inter_parallel,
                                   graph_options=tf.GraphOptions(enable_bfloat16_sendrecv=FLAGS.use_bfloat16_transfer),
				   gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.00001)
		                  )
    if FLAGS.enable_jit:
        server_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    server = tf.train.Server(cluster, job_name="ps", task_index=ps_index, config=server_config)
    print(datetime.now(), "ps %d: listening ..." % ps_index)
    server.join()

def start_ps():
    print(datetime.now(), "starting ps")
    processes = [multiprocessing.Process(target=ps, args=[i]) for i in range(FLAGS.ps_per_host)]
    [p.start() for p in processes]

    app_id = os.environ.get("APP_ID")

    if not gfile.Exists(FLAGS.log_dir):
        gfile.MakeDirs(FLAGS.log_dir)

    # write signal files
    signal_path = FLAGS.model_dir + "/" + get_app_id() + "/signals"
    signal_file = signal_path + "/ps_host_%d.ready" % FLAGS.task_index
    if not gfile.Exists(signal_path):
        gfile.MakeDirs(signal_path)
    
    print(datetime.now(), "start to write ps ready signal file")
    with gfile.Open(signal_file, "w") as f:
        f.write("ready")
    
    ps_ips, _ = get_machine_list()

    print(datetime.now(), "wait for all ps ready")
    while True:
        files = gfile.ListDirectory(signal_path)
        if len(files) != len(ps_ips):
            print(datetime.now(), "ready ps:", ",".join(files))
            time.sleep(30)
        else:
            print(datetime.now(), "all ps ready:", ",".join(files))
            break
    
    return processes

def start_worker():
    print(datetime.now(), "starting workers")
    processes = [multiprocessing.Process(target=train_dist, args=[index]) for index in range(0, FLAGS.worker_per_host)]
    [p.start() for p in processes]
    
    return processes

def main(_):
    multiprocessing.set_start_method('spawn')

    begin = time.time()

    if FLAGS.is_validation:
        if not gfile.Exists(FLAGS.output_dir):
            gfile.MakeDirs(FLAGS.output_dir)
        evalute()
    else:
        if FLAGS.ps_hosts == "" or (FLAGS.ps_hosts != "" and FLAGS.job_name == "ps"):
            ps_processes = start_ps()

        else:
            ps_processes = []
        
        if FLAGS.ps_hosts == "" or (FLAGS.ps_hosts != "" and FLAGS.job_name == "worker"):
            worker_processes = start_worker()
        else:
            worker_processes = []

        print(datetime.now(), "start to monitor all child processes")
        #monitor_children(ps_processes, worker_processes)
        print(datetime.now(), "Finish in %.3f seconds" % (time.time() - begin))
        time.sleep(1000)

if __name__ == "__main__":
    tf.app.run()
