import tensorflow as tf
import random
from tensorflow.python.user_ops import user_ops
import sys

class AdsDataReader:

    def __init__(self, filename_queue):
        with tf.variable_scope("ads_dnn_input"):
            #filename_queue = tf.train.string_input_producer(data_files)
            _, batch_example = tf.TFRecordReader().read(filename_queue)
            parsed_batch = tf.parse_single_example(batch_example, features={'value' : tf.FixedLenFeature([], tf.string)})
            int_tensor = tf.decode_raw(parsed_batch['value'], tf.int32)

        self._int_tensor = int_tensor

    @property
    def int_tensor(self):
        return self._int_tensor


class AdsDnnModel:
    def __init__(self, filename_queue, feature_dimensions, batch_size, embedding_count, layer1_count, layer2_count, label_count, learning_rate, ps_count, use_reduce_sum, optimizer):
        with tf.variable_scope("ads_dnn_input"):
            _, batch_example = tf.TFRecordReader().read_up_to(filename_queue, batch_size)

        with tf.variable_scope("ads_dnn_model"):
            if sys.platform == "win32":
                sample_labels, sample_weights, position_indices, position_values, position_shapes, term_indices, term_values, term_shapes = user_ops.parse_ads_dnn_samples(batch_example, len(feature_dimensions) - 1)
            else:
                ads_dnn_module = tf.load_op_library('./ads_dnn.so')
                sample_labels, sample_weights, position_indices, position_values, position_shapes, term_indices, term_values, term_shapes = ads_dnn_module.parse_ads_dnn_samples(batch_example, len(feature_dimensions) - 1)

            pos_sp_tensor = tf.SparseTensor(position_indices, position_values, position_shapes)
            term_sp_tensor = tf.SparseTensor(term_indices, term_values, term_shapes)
            #y = tf.reshape(tf.bitcast(tf.slice(int_tensor, [0], [batch_size]), tf.float32), [-1, 1])
            #sample_weights = tf.bitcast(tf.slice(int_tensor, [batch_size], [batch_size]), tf.float32)
            #int64_part = tf.bitcast(tf.reshape(tf.slice(int_tensor, [batch_size * 2], [-1]), [-1, 2]), tf.int64)

            #pos_sp_indices= tf.reshape(tf.slice(int64_part, [batch_size], [batch_size * 2]), [-1, 2])
            #pos_sp_ids = tf.slice(int64_part, [0], [batch_size])
            #pos_sp_shapes = tf.constant([batch_size, 1], dtype=tf.int64)
            #pos_sp_tensor = tf.SparseTensor(pos_sp_indices, pos_sp_ids, pos_sp_shapes)

            #term_sp_part = tf.reshape(tf.slice(int64_part, [batch_size * 3 + 2], [-1]), [3, -1])
            #term_sp_indices = tf.reshape(tf.slice(term_sp_part, [1, 0], [-1, -1]), [-1, 2])
            #term_sp_ids = tf.reshape(tf.slice(term_sp_part, [0, 0], [1, -1]), [-1])
            #term_sp_shapes = tf.slice(int64_part, [batch_size * 3], [2])
            #term_sp_tensor = tf.SparseTensor(term_sp_indices, term_sp_ids, term_sp_shapes)

            feature_group_count = len(feature_dimensions)
            term_dimension_count = sum(feature_dimensions)
            # assign one default feature id for empty group
            term_dimension_count += len(feature_dimensions) + 1  # feature id start from 0
            print("total term feature count:", term_dimension_count)
            variable_initializer = tf.random_uniform_initializer(-0.05, 0.05)

            dense_tensor_size = embedding_count * (feature_group_count - 1) * layer1_count + layer1_count * layer2_count + layer2_count * label_count + (feature_dimensions[0] + 1) * label_count

            #term_embedding_ps_weight = batch_size * embedding_count * 100
            #dense_ps_weight = dense_tensor_size
            #ps_weight_sum = term_embedding_ps_weight + dense_ps_weight

            #term_embedding_ps_count = int(term_embedding_ps_weight * ps_count / ps_weight_sum) + 1
            #dense_ps_count = int(dense_ps_weight * ps_count / ps_weight_sum) + 1
            #if term_embedding_ps_count > ps_count:
            #    term_embedding_ps_count = ps_count
            #if dense_ps_count > ps_count:
            #    dense_ps_count = ps_count
            #print("term embedding ps count:", term_embedding_ps_count, "dense ps count:", dense_ps_count)

            embedding_out_offset = 0
            embedding_out_size = embedding_count * (feature_group_count - 1) * layer1_count
            embedding_out_shape = [embedding_count * (feature_group_count - 1), layer1_count]

            l1l2_offset = embedding_out_offset + embedding_out_size
            l1l2_size = layer1_count * layer2_count
            l1l2_shape = [layer1_count, layer2_count]

            final_offset = l1l2_offset + l1l2_size
            final_size = layer2_count * label_count
            final_shape = [layer2_count, label_count]

            pos_offset = final_offset + final_size
            pos_size = (feature_dimensions[0] + 1) * label_count
            pos_shape = [(feature_dimensions[0] + 1), label_count]

            W_dense_part = tf.get_variable("W_dense_part", [dense_tensor_size], partitioner=tf.fixed_size_partitioner(ps_count), initializer=variable_initializer)

            W_term_embeddings = tf.get_variable("W_term_embeddings", [term_dimension_count, embedding_count], partitioner=tf.fixed_size_partitioner(ps_count), initializer=variable_initializer)
            #W_embeddings_out = tf.get_variable("W_embedding_out", [embedding_count*(feature_group_count-1), layer1_count], partitioner=tf.fixed_size_partitioner(embedding_out_ps_count), initializer=variable_initializer)
            #W_l1_l2 = tf.get_variable("W_l1_l2", [layer1_count, layer2_count], partitioner=tf.fixed_size_partitioner(l1_l2_ps_count), initializer=variable_initializer)

            #W_final = tf.get_variable("W_final", [layer2_count, label_count], initializer=variable_initializer)
            #W_pos = tf.get_variable("W_pos", [feature_dimensions[0], label_count], initializer=variable_initializer)
            global_step = tf.Variable(0, name="global_step", trainable=False)

            with tf.device("/cpu:0"):
                term_embeddings = tf.nn.embedding_lookup_sparse(W_term_embeddings, sp_ids=term_sp_tensor, sp_weights=None, combiner="sum")
                term_embeddings = tf.reshape(term_embeddings, [-1, embedding_count*(feature_group_count-1)])

                W_embeddings_out = tf.reshape(tf.slice(W_dense_part, [embedding_out_offset], [embedding_out_size]), embedding_out_shape)
                W_l1_l2 = tf.reshape(tf.slice(W_dense_part, [l1l2_offset], [l1l2_size]), l1l2_shape)
                W_final = tf.reshape(tf.slice(W_dense_part, [final_offset], [final_size]), final_shape)
                W_pos = tf.reshape(tf.slice(W_dense_part, [pos_offset], [pos_size]), pos_shape)

                embedding_out = tf.matmul(term_embeddings, W_embeddings_out)
                layer1_output = tf.nn.relu(embedding_out)
                layer2_input = tf.matmul(layer1_output, W_l1_l2)
                layer2_output = tf.nn.relu(layer2_input)
                layer3_input1 = tf.matmul(layer2_output, W_final)

                layer3_input2 = tf.matmul(tf.to_float(tf.sparse_to_indicator(pos_sp_tensor, feature_dimensions[0] + 1)), W_pos)
                #layer3_input2 = tf.nn.embedding_lookup_sparse(W_pos, sp_ids=pos_sp_tensor, sp_weights=None, combiner="sum")
                pred_pre_sigmoid = tf.add(layer3_input1, layer3_input2)
                pred = tf.nn.sigmoid(pred_pre_sigmoid)

            unweighted_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_pre_sigmoid, targets=sample_labels)
            #unweighted_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_pre_sigmoid, labels=sample_labels)
            final_loss = tf.multiply(sample_weights, unweighted_loss)

            if use_reduce_sum:
                loss_fn = tf.reduce_sum(final_loss)
            else:
                loss_fn = tf.reduce_mean(final_loss)

            print("Optimizer type:", optimizer)
            if optimizer == "AdaGrad":
                train_op = tf.train.AdagradOptimizer(learning_rate).minimize(loss_fn, global_step=global_step)
            elif optimizer == "RMSProp":
                train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(loss_fn, global_step=global_step)
            elif optimizer == "FTRL":
                train_op = tf.train.FtrlOptimizer(learning_rate=learning_rate, l1_regularization_strength=0,l2_regularization_strength=0).minimize(loss_fn, global_step=global_step)
            else:
                train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_fn, global_step=global_step)
            self._labels = sample_labels
            self._weights = sample_weights
            self._prediction = pred
            self._global_step = global_step
            self._loss_fn = loss_fn
            self._train_op = train_op
            self._final = W_final
            self._w_l1_l2 = W_l1_l2
            self._layer3_input1 = layer3_input1
            self._layer3_input2 = layer3_input2
            self._layer2_input = layer2_input
            self._layer1_output = layer1_output
            self._term_embeddings = term_embeddings
            self._w_pos = W_pos
            self._pos_sp_tensor = pos_sp_tensor
            self._layer2_output = layer2_output
            self._embedding_out = embedding_out
            self._unweighted_loss = unweighted_loss
            self._final_loss = final_loss

            self._term_shapes = term_shapes
    @property
    def term_shapes(self):
        return self._term_shapes

    @property
    def final_loss(self):
        return self._final_loss
    @property
    def unweighted_loss(self):
        return self._unweighted_loss
    @property
    def embedding_out(self):
        return self._embedding_out
    @property
    def layer2_output(self):
        return self._layer2_output
    @property
    def pos_tensor(self):
        return self._pos_sp_tensor
    @property
    def pos(self):
        return self._w_pos

    @property
    def final(self):
        return self._final
    @property
    def w_l1_l2(self):
        return self._w_l1_l2

    @property
    def layer3_input1(self):
        return self._layer3_input1
    @property
    def layer3_input2(self):
        return self._layer3_input2
    @property
    def layer2_input(self):
        return self._layer2_input
    @property
    def layer1_output(self):
        return self._layer1_output
    @property
    def term_embeddings(self):
        return self._term_embeddings
    @property
    def labels(self):
        return self._labels

    @property
    def weights(self):
        return self._weights

    @property
    def prediction(self):
        return self._prediction

    @property
    def global_step(self):
        return self._global_step

    @property
    def loss_fn(self):
        return self._loss_fn

    @property
    def train_op(self):
        return self._train_op
