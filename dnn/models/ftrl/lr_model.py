from tensorflow.python.user_ops import user_ops
from tensorflow.python.framework import ops
import tensorflow as tf
import platform
import hash_embedding_ops
import raw_key_ops
import util
import distributed_runner

tf.app.flags.DEFINE_integer("hash_bucket_size", 500000000, "hash bucket size")
tf.app.flags.DEFINE_float("l1", 0, "l1")
tf.app.flags.DEFINE_float("l2", 0, "l2")
tf.app.flags.DEFINE_float("poisson", 0.1, "poisson feature inclusion")
tf.app.flags.DEFINE_boolean("output_optimizer_slots", True, "output optimizer slots(for example FTRL n, z) or not")
tf.app.flags.DEFINE_float("learning_rate", 0.02, "learning rate")
tf.app.flags.DEFINE_string("optimizer", "FTRL", "optimizer for training.")
tf.app.flags.DEFINE_integer("nloop", 1, "mini batch count inside one graph")

tf.app.flags.DEFINE_boolean("disable_sparse_grad_unique", True, "disable sparse gradients unique")

FLAGS = tf.app.flags.FLAGS

class AdsLrModel(distributed_runner.DistributedRunner):
    def __init__(self):
        super(AdsLrModel, self).__init__()

    def create_model(self, filename_queue):
        optimizer = FLAGS.optimizer
        hash_bucket_size = FLAGS.hash_bucket_size
        nloop = FLAGS.nloop
        with_dependency = FLAGS.with_dependency
        poisson = FLAGS.poisson
        if FLAGS.is_validation:
            poisson = 0.0

        ps_count = self.get_cluster_ps_count()

        with tf.variable_scope("ads_lr_input"):
            reader = tf.TFRecordReader()

        with tf.variable_scope("ads_lr_model"):
            hash_table = tf.get_variable("HashTable", [hash_bucket_size, 3],
                                         initializer=tf.zeros_initializer,
                                         dtype=tf.int32, trainable=False,
                                         partitioner=tf.fixed_size_partitioner(ps_count))
            W_weights = tf.get_variable("W_weights", [hash_bucket_size],
                                        initializer=tf.zeros_initializer, dtype=tf.float32,
                                        partitioner=tf.fixed_size_partitioner(ps_count))

            global_step = tf.Variable(0, name="global_step", trainable=False)

            # --------------------------------- debug
            hash_table_list = list(hash_table)
            for v in hash_table_list:
                print("name:", v.name, "device:", v.device, "shape:", v.get_shape())
            # --------------------------------- debug
            if optimizer == "AdaGrad":
                optimizer_op = tf.train.AdagradOptimizer(FLAGS.learning_rate)
            elif optimizer == "FTRL":
                optimizer_op = tf.train.FtrlOptimizer(learning_rate=FLAGS.learning_rate,
                                                      l1_regularization_strength=FLAGS.l1,
                                                      l2_regularization_strength=FLAGS.l2)
            else:
                raise ValueError("Unrecognized optimizer type" + optimizer)

            # patch optimizer _apply_sparse_duplicate_indices
            if FLAGS.disable_sparse_grad_unique:
                optimizer_op._apply_sparse_duplicate_indices = optimizer_op._apply_sparse
            def one_mini_batch(batch_index, dependencies):
                # read one minibatch data
                with ops.control_dependencies(dependencies):
                    _, batch_example = reader.read_up_to(filename_queue, FLAGS.batch_size)
                    sample_labels, sample_weights, sample_guids, feature_indices, feature_values, feature_shape = raw_key_ops.parse_lr_samples(batch_example)
                    sample_labels = tf.reshape(sample_labels, [-1, 1])
                    sample_weights = tf.reshape(sample_weights, [-1, 1])
                    sample_guids = tf.reshape(sample_guids, [-1, 1])

                with tf.device("/cpu:0"):
                    before_sigmoid = hash_embedding_ops.hash_embedding_lookup_sparse(hash_table, W_weights, feature_indices, feature_values, feature_shape, poisson=poisson)
                    before_sigmoid = tf.reshape(before_sigmoid, [-1, 1])

                    pred = tf.nn.sigmoid(before_sigmoid)

                    unweighted_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=before_sigmoid, labels=sample_labels)
                    final_loss = tf.multiply(sample_weights, unweighted_loss)

                    if FLAGS.use_reduce_sum:
                        loss_fn = tf.reduce_sum(final_loss)
                    else:
                        loss_fn = tf.reduce_mean(final_loss)

                    train_op = optimizer_op.minimize(loss_fn, global_step=global_step)
                    if with_dependency:
                        return [loss_fn], train_op, pred, loss_fn, sample_labels, sample_weights, sample_guids
                    else:
                        return [], train_op, pred, loss_fn, sample_labels, sample_weights, sample_guids

            dependencies = []
            train_op_array = []
            predictions = []
            loss_fns = []
            sample_labels_array = []
            sample_weights_array = []
            sample_guids_array = []
            for i in range(nloop):
                dependency, train_op, prediction, loss_fn, sample_labels, sample_weights, sample_guids = one_mini_batch(i, None if i == 0 or not with_dependency else dependencies[-1])
                dependencies.append(dependency)
                train_op_array.append(train_op)
                predictions.append(prediction)
                loss_fns.append(loss_fn)
                sample_labels_array.append(sample_labels)
                sample_weights_array.append(sample_weights)
                sample_guids_array.append(sample_guids)

            train_ops = tf.group(*train_op_array)

            weight_save_ops, weight_restore_ops = util.save_model_for_raw_key(
                    FLAGS.init_model_dir, FLAGS.model_dir, "lr_weights", hash_table, W_weights, optimizer_op, FLAGS.output_optimizer_slots)

            # define custom saver for raw key
            self._save_op = tf.tuple(weight_save_ops)
            self._restore_op = tf.tuple(weight_restore_ops) if weight_restore_ops else []

            self._labels = sample_labels_array
            self._weights = sample_weights_array
            self._train_op = train_ops
            self._predictions = predictions
            self._loss_fn = loss_fns
            self._global_step = global_step
            self._guids = sample_guids_array