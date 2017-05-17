from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
import tensorflow as tf

import platform
import os

raw_key_ops_so = "_raw_key_ops.so"

if platform.system() == "Windows":
    raw_key_ops_so = "_raw_key_ops.dll"

raw_key_module = tf.load_op_library(os.path.join(os.path.dirname(__file__), raw_key_ops_so))

def woodblocks_unique(woodblocks_keys):
    return raw_key_module.woodblocks_unique(woodblocks_keys)

def filter(data, target):
    return raw_key_module.filter(data, target)

def hash_to_id(hash_table, woodblocks_keys, poisson):
    return raw_key_module.hash_to_id(hash_table, woodblocks_keys, poisson)

def modify_first_dim(shape1, shape2):
    return raw_key_module.modify_first_dim(shape1, shape2)

def murmur32(woodblocks_keys):
    return raw_key_module.murmur32(woodblocks_keys)

def variables_save(model_path, model_name, partition_no, variables):
    return raw_key_module.variables_save(model_path, model_name, partition_no, variables)

def variables_restore(model_path, model_name, partition_count, partition_no, variables):
    return raw_key_module.variables_restore(model_path, model_name, partition_count, partition_no, variables)

def hash_variables_save(model_path, model_name, partition_no, hash_table, variables):
    return raw_key_module.hash_variables_save(model_path, model_name, partition_no, hash_table, variables)

def hash_variables_restore(model_path, model_name, partition_count, partition_no, hash_table, variables):
    return raw_key_module.hash_variables_restore(model_path, model_name, partition_count, partition_no, hash_table, variables)

def parse_dnn_samples(batch_samples, feature_set, position_feature):
    return raw_key_module.parse_dnn_samples(batch_samples, feature_set, position_feature)

def parse_lr_samples(batch_samples):
    return raw_key_module.parse_lr_samples(batch_samples)

def fast_dynamic_partition(data, partitions, np):
    return raw_key_module.fast_dynamic_partition(data, partitions, np)

@ops.RegisterGradient("FastDynamicPartition")
def _FastDynamicPartitionGrads(op, *grads):
  """Gradients for FastDynamicPartition. copy from DynamicPartition"""
  data = op.inputs[0]
  indices = op.inputs[1]
  num_partitions = op.get_attr("num_partitions")

  prefix_shape = array_ops.shape(indices)
  original_indices = array_ops.reshape(
      math_ops.range(math_ops.reduce_prod(prefix_shape)), prefix_shape)
  partitioned_indices = data_flow_ops.dynamic_partition(
      original_indices, indices, num_partitions)
  reconstructed = data_flow_ops.dynamic_stitch(partitioned_indices, grads)
  reconstructed = array_ops.reshape(reconstructed, array_ops.shape(data))
  return [reconstructed, None]

