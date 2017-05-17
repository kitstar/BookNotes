import multiprocessing
import os
import sys
import tensorflow as tf
import time
from datetime import datetime
from tensorflow.python.client import timeline
from tensorflow.python.platform import gfile
from util import print_model

# IO arguments
tf.app.flags.DEFINE_string("data_dir", "data", "File path of input data.")
tf.app.flags.DEFINE_string("log_dir", ".", "Directory path storing log files.")
tf.app.flags.DEFINE_string("init_model_dir", "", "initial model dir")
tf.app.flags.DEFINE_string("model_dir", "model", "model output dir")
tf.app.flags.DEFINE_string("output_dir", "output", "Directory path of evalutation output.")
tf.app.flags.DEFINE_string("output_name", "prev.tsv", "File name of evalutation output.")

# Distributed arguments
tf.app.flags.DEFINE_integer("batch_size", 400, "batch size")
tf.app.flags.DEFINE_integer("iteration", 1, "num of epochs before stopping")
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
tf.app.flags.DEFINE_boolean("use_bfloat16_transfer", True, "Use bfloat16 to reduce network traffic")
tf.app.flags.DEFINE_float("success_ratio", 0.95, "master work will exit after success_ratio workers finished")
tf.app.flags.DEFINE_integer("session_run_timeout", 2000, "session single step timeout in ms")
tf.app.flags.DEFINE_integer("wait_for_exit_timeout", 1800, "wait for other workers to finish before force exiting")
tf.app.flags.DEFINE_integer("stop_after_timeout_count", 100, "stop training after N timeout count")
tf.app.flags.DEFINE_boolean("max_checkpoint", 1, "Keep max checkpoint count")
tf.app.flags.DEFINE_integer("ps_per_host", 1, "parameter server count per host")
tf.app.flags.DEFINE_float("exit_after_per_host_failure", 0.5, "exit after per host failure")
tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "allow soft placement")
tf.app.flags.DEFINE_boolean("enable_jit", False, "enable XLA jit")
tf.app.flags.DEFINE_integer("max_wait_secs", 7200, "wait for master worker")

tf.app.flags.DEFINE_integer("report_progress_step", 100, "")
tf.app.flags.DEFINE_boolean("log_device_placement", False, "Print placement of variables and operations.")
tf.app.flags.DEFINE_integer("log_timeline_step", 1000, "log time each N steps")

tf.app.flags.DEFINE_integer("max_step", 0, "max step to stop")

tf.app.flags.DEFINE_boolean("is_validation", False, "is validation job")

# make it as common flag
tf.app.flags.DEFINE_integer("label_count", 1, "label count")
tf.app.flags.DEFINE_boolean("use_reduce_sum", True, "use reduce sum or reduce mean")

tf.app.flags.DEFINE_boolean("with_dependency", True, "parallel or serialize n minibatches inside one graph")

FLAGS = tf.app.flags.FLAGS

class DistributedRunner(object):
    def __init__(self):
        self._debug_tensors = None

    def get_app_id(self):
        app_id = os.environ.get("APP_ID")
        if not app_id:
            app_id = "None"
        return app_id

    def custom_global_variables_initializer(self):
        blacklist = ["custom_restore_signal"]
        return tf.group(*[x.initializer for x in list(filter(lambda x: x.op.name not in blacklist, tf.global_variables()))])

    def evalute(self):
        # Prepare graph and configuration
        data_files = list(filter(lambda x: not gfile.IsDirectory(x), [os.path.join(FLAGS.data_dir, x) for x in gfile.ListDirectory(FLAGS.data_dir)]))
        filename_queue = tf.train.string_input_producer(data_files, num_epochs=1)
        #model = create_graph(filename_queue)
        self.create_model(filename_queue)
        init_op = tf.global_variables_initializer()
        local_variables_init_op = tf.local_variables_initializer()
        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=FLAGS.log_device_placement)
        output_path = os.path.join(FLAGS.output_dir, FLAGS.output_name)

        with tf.Session(config=sess_config) as sess, open(output_path, "w") as filew:
            sess.run([init_op, local_variables_init_op])

            print("restore from model: ", sess.run(self.restore_op))

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

                    _pred, _labels, _weights, _loss = sess.run([self.predictions, self.labels, self.weights, self.loss_fn])
                    end = time.time()
                    #print("loss:", _loss)

                    # Update counters
                    local_step += 1
                    graph_comptime += end - begin
                    if local_step % FLAGS.report_progress_step == 0:
                        print("local_step: %d, mean_step_time: %.3f, validation_time: %.2f" % (local_step, graph_comptime/local_step, end-evaluation_start))

                    # Output result
                    for j in range(len(_pred)):
                        for i in range(len(_pred[j])):
                            filew.write(str(_pred[j][i][0]) + '\t' + str(_labels[j][i][0]) + '\t' + str(_weights[j][i][0]) + '\n')
            except tf.errors.OutOfRangeError:
                print("Reach EOF of data")

            # Stop all threads
            coord.request_stop()
            coord.join(threads)

    def train_dist(self, process_index):
        # redirect stdout stderr to files
        task_index = FLAGS.task_index * FLAGS.worker_per_host + process_index
        log_file_name = os.path.join(FLAGS.log_dir, "worker_%d.log" % task_index)
        sys.stdout = open(log_file_name, 'w', 1)
        sys.stderr = sys.stdout

        ps_servers, worker_servers = self.get_cluster_servers()

        print(datetime.now(), "Identify role, host worker:%d started, task_index:%d, Pid:%d" % (process_index, task_index, os.getpid()))
        cluster = tf.train.ClusterSpec({"ps": ps_servers, "worker": {task_index: worker_servers[task_index]}})
        is_chief = task_index == 0

        ps_count = len(cluster.job_tasks("ps"))

        # Start GRPC server
        print(datetime.now(), "Start GRPC server")
        sess_config = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                     log_device_placement=FLAGS.log_device_placement,
                                     intra_op_parallelism_threads=FLAGS.wk_intra_parallel,
                                     inter_op_parallelism_threads=FLAGS.wk_inter_parallel,
                                     graph_options=tf.GraphOptions(enable_bfloat16_sendrecv=FLAGS.use_bfloat16_transfer))
        if FLAGS.enable_jit:
            sess_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        server = tf.train.Server(cluster, job_name="worker", task_index=task_index, config = sess_config)

        worker_device = "/job:worker/task:%d" % task_index

        data_files = list(filter(lambda x: not gfile.IsDirectory(x), [os.path.join(FLAGS.data_dir, x) for x in gfile.ListDirectory(FLAGS.data_dir)]))
        print(datetime.now(), "data files:", data_files)
        if len(data_files) % FLAGS.worker_per_host != 0:
            sys.exit("Exit due to len(%s) mod %d != 0" % (data_files, FLAGS.worker_per_host))
        file_count_per_process = int(len(data_files) / FLAGS.worker_per_host)
        print(datetime.now(), "split file to %d workers, each worker %d files" % (FLAGS.worker_per_host, file_count_per_process))

        # sort by file name to be deterministic
        data_files.sort()
        process_data_files = data_files[file_count_per_process * process_index : file_count_per_process * (process_index + 1)]
        print(datetime.now(), "process data files:", process_data_files)

        with tf.device(worker_device):
            filename_queue = tf.train.string_input_producer(process_data_files, num_epochs=FLAGS.iteration)

        # Create graph
        print("%s: Create graph" % str(datetime.now()))
        device_setter = tf.train.replica_device_setter(worker_device=worker_device, cluster=cluster)
        with tf.device(device_setter):
            finish_count = tf.Variable(0, name="finish_count", trainable=False)
            exit_signal = tf.Variable(0, name="exit_signal", trainable=False)
            custom_restore_signal = tf.Variable(0, name="custom_restore_signal", trainable=False)
            worker_finish_op = finish_count.assign_add(1, use_locking=False)
            exit_signal_op = exit_signal.assign_add(1, use_locking=False)
            #model = create_graph(filename_queue)
            self.create_model(filename_queue)

        report_uninitialized_op = tf.report_uninitialized_variables(tf.global_variables())

        global_var_init_op = self.custom_global_variables_initializer()

        def restore_or_initialize_variables(sess):
            # Always initialize all global variables first except custom_restore_signal, custom_restore_signal
            # is used for blocking model ready_op
            print(datetime.now(), "start to initialize all variables except custom restore signal")
            sess.run(global_var_init_op)
            print(datetime.now(), "start to restore model from files")
            print(datetime.now(), sess.run(self.restore_op))
            print(datetime.now(), "start to initialize custom restore signal")
            sess.run(custom_restore_signal.initializer)
            print_model(sess)

        print_model(None)

        sv = tf.train.Supervisor(is_chief=is_chief,
                                 init_op=None,
                                 init_fn=restore_or_initialize_variables,
                                 local_init_op=tf.local_variables_initializer(),
                                 logdir=FLAGS.model_dir,
                                 global_step=self.global_step,
                                 saver=None, # Disable the saver of Supervisor which separates checkpoint files cross machines!
                                 save_model_secs=None, # if FLAGS.save_model_secs<=0 else FLAGS.save_model_secs,
                                 summary_writer=None)
        with sv.prepare_or_wait_for_session(master=server.target, config=sess_config, start_standard_services=True, max_wait_secs=FLAGS.max_wait_secs) as sess:
            print(datetime.now(), "Create session success")

            local_step = 0
            graph_comptime = 0.0
            train_start = time.time()
            local_host_loss = 0
            report_progress_sum_loss = 0
            last_global_step = sess.run(self.global_step)
            last_timestamp = time.time()

            sess_timeout_count = 0
            is_eof = False

            try:
                while not sv.should_stop() and sess_timeout_count < FLAGS.stop_after_timeout_count:
                    if FLAGS.max_step != 0 and local_step > FLAGS.max_step:
                        print(datetime.now(), "max_step is defined, stop after", FLAGS.max_step)
                        break
                    begin = time.time()
                    try:
                        #run_metadata = tf.RunMetadata()
                        _, step_loss, step_count = sess.run([self.train_op, self.loss_fn, self.global_step], options=tf.RunOptions(timeout_in_ms=FLAGS.session_run_timeout)) #, trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_metadata)
                        #_, step_loss, step_count, _label, _weight, _pred = sess.run([model.train_op, model.loss_fn, model.global_step, model.labels, model.weights, model.prediction], options=tf.RunOptions(timeout_in_ms=FLAGS.session_run_timeout)) #, trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_metadata)
                    except tf.errors.DeadlineExceededError:
                        print(datetime.now(), "session run deadline exceed:", FLAGS.session_run_timeout, "session timeout count:", sess_timeout_count)
                        sess_timeout_count += 1
                        continue
                    end = time.time()
                    loss_list = list(step_loss)
                    local_host_loss += sum(loss_list)
                    report_progress_sum_loss += sum(loss_list)

                    # Update counters
                    local_step += len(loss_list)
                    graph_comptime += end - begin
                    train_time = end - train_start
                    if int(local_step / len(loss_list)) % int(FLAGS.report_progress_step / len(loss_list)) == 0:
                        print("%s: global_step: %d, local_step:%d, accu_mean_loss: %.3f, report_interval_mean_loss: %.3f, mean_step_time: %.3f, train_time:%.2f, cluster_throughput:%.2f" % (str(datetime.now()), step_count, local_step, local_host_loss / local_step / FLAGS.batch_size , report_progress_sum_loss / FLAGS.report_progress_step / FLAGS.batch_size, (end - last_timestamp) / FLAGS.report_progress_step, train_time, (step_count - last_global_step) * FLAGS.batch_size / (end - last_timestamp)))
                        last_global_step = step_count
                        last_timestamp = end
                        report_progress_sum_loss = 0
                        #if True:
                        #    trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                        #    trace_file = open(os.path.join(FLAGS.log_dir, "timeline-%d-%d.ctf.json" % (task_index, local_step)), 'w')
                        #    trace_file.write(trace.generate_chrome_trace_format())
                        #    trace_file.close()
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
                print(datetime.now(), sess.run(self.save_op))
                print(datetime.now(), "Save model done, exit")
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

    def monitor_children(self, ps_processes, worker_processes):
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
                        [p.terminate() for p in ps_processes]
                        [p.terminate() for p in worker_processes]
                        sys.exit("Exit due to ps failure, kill all children")

                    print(datetime.now(), "ps inner_index:%d Pid:%d exit with code:%d, runtime:%3.f seconds" % (id, alive_ps[id].pid, exit_code, time.time() - begin))
                    alive_ps.pop(id)

            alive_worker_ids = list(alive_worker.keys())
            for id in alive_worker_ids:
                if not alive_worker[id].is_alive():
                    exit_code = alive_worker[id].exitcode
                    print(datetime.now(), "worker inner_index:%d Pid:%d exit with code:%d, runtime:%3.f seconds" % (id, alive_worker[id].pid, exit_code, time.time() - begin))
                    alive_worker.pop(id)

                    if exit_code != 0:
                        worker_failure_count += 1
                        if len(worker_processes) * FLAGS.exit_after_per_host_failure <= worker_failure_count:
                            [p.terminate() for p in ps_processes]
                            [p.terminate() for p in worker_processes]
                            sys.exit("Exit due to too many failures, kill all children")
                    if id == 0 and FLAGS.task_index == 0:
                        print(datetime.now(), "master worker exit, exit process, kill all alive children")
                        [p.terminate() for p in ps_processes]
                        [p.terminate() for p in worker_processes]
                        if exit_code != 0:
                            print(datetime.now(), "master worker failed")
                            sys.exit("master worker failed")
                        return

    def get_machine_list(self):
        ps_hosts = FLAGS.ps_hosts
        worker_hosts = FLAGS.worker_hosts
        if ps_hosts == "":
            ps_hosts = worker_hosts
        ps_ips = ps_hosts.strip().split(",")
        worker_ips = worker_hosts.strip().split(",")
        return ps_ips, worker_ips

    def get_cluster_ps_count(self):
        ps_servers, _ = self.get_cluster_servers()
        return len(ps_servers)

    def get_cluster_servers(self):
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

    def ps(self, inner_host_index):
        ps_servers, worker_servers = self.get_cluster_servers()

        cluster = tf.train.ClusterSpec({"ps": ps_servers, "worker": worker_servers})
        ps_index = FLAGS.task_index * FLAGS.ps_per_host + inner_host_index

        log_file_name = os.path.join(FLAGS.log_dir, "ps_%d.log" % ps_index)
        sys.stdout = open(log_file_name, 'w', 1)
        sys.stderr = sys.stdout

        server_config = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                       log_device_placement=FLAGS.log_device_placement,
                                       intra_op_parallelism_threads=FLAGS.ps_intra_parallel,
                                       inter_op_parallelism_threads=FLAGS.ps_inter_parallel,
                                       graph_options=tf.GraphOptions(enable_bfloat16_sendrecv=FLAGS.use_bfloat16_transfer))
        if FLAGS.enable_jit:
            server_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        server = tf.train.Server(cluster, job_name="ps", task_index=ps_index, config=server_config)
        print(datetime.now(), "ps %d: listening ..." % ps_index)
        server.join()

    def start_ps(self):
        print(datetime.now(), "starting ps")
        processes = [multiprocessing.Process(target=self.ps, args=[i]) for i in range(FLAGS.ps_per_host)]
        [p.start() for p in processes]

        app_id = self.get_app_id()

        if not gfile.Exists(FLAGS.log_dir):
            gfile.MakeDirs(FLAGS.log_dir)

        # write signal files
        signal_path = FLAGS.model_dir + "/" + app_id + "/signals"
        signal_file = signal_path + "/ps_host_%d.ready" % FLAGS.task_index
        if not gfile.Exists(signal_path):
            gfile.MakeDirs(signal_path)

        print(datetime.now(), "start to write ps ready signal file")
        with gfile.Open(signal_file, "w") as f:
            f.write("ready")

        ps_ips, _ = self.get_machine_list()

        print(datetime.now(), "wait for all ps ready")
        if not app_id:
            # local env, wait for 5 seconds
            time.sleep(5)
            return processes

        while True:
            files = gfile.ListDirectory(signal_path)
            if len(files) != len(ps_ips):
                print(datetime.now(), "ready ps:", ",".join(files))
                time.sleep(30)
            else:
                print(datetime.now(), "all ps ready:", ",".join(files))
                break

        return processes

    def start_worker(self):
        print(datetime.now(), "starting workers")
        processes = [multiprocessing.Process(target=self.train_dist, args=[index]) for index in range(0, FLAGS.worker_per_host)]
        [p.start() for p in processes]

        return processes

    def create_model(self, filename_queue):
        raise NotImplementedError()

    def run(self):
        multiprocessing.set_start_method("spawn")
        begin = time.time()

        if FLAGS.is_validation:
            if not gfile.Exists(FLAGS.output_dir):
                gfile.MakeDirs(FLAGS.output_dir)
            self.evalute()
        else:
            if FLAGS.ps_hosts == "" or (FLAGS.ps_hosts != "" and FLAGS.job_name == "ps"):
                ps_processes = self.start_ps()
            else:
                ps_processes = []

            if FLAGS.ps_hosts == "" or (FLAGS.ps_hosts != "" and FLAGS.job_name == "worker"):
                worker_processes = self.start_worker()
            else:
                worker_processes = []

            print(datetime.now(), "start to monitor all child processes")
            self.monitor_children(ps_processes, worker_processes)
            print(datetime.now(), "Finish in %.3f seconds" % (time.time() - begin))

    @property
    def global_step(self):
        return self._global_step

    @property
    def loss_fn(self):
        return self._loss_fn

    @property
    def train_op(self):
        return self._train_op

    @property
    def labels(self):
        return self._labels

    @property
    def weights(self):
        return self._weights

    @property
    def guids(self):
        return self._guids

    @property
    def predictions(self):
        return self._predictions

    @property
    def save_op(self):
        return self._save_op

    @property
    def restore_op(self):
        return self._restore_op

    @property
    def debug_tensors(self):
        return self._debug_tensors
