import os
import tensorflow as tf
import time

from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.python.ops import variables
from tensorflow.python.framework import ops
import raw_key_ops

def print_model(sess):
    print("Local variables are:")
    for v in tf.local_variables():
        print("Parameter:", v.name, "device:", v.device, "shape:", v.get_shape())
    print("Global variables are:")
    for v in tf.global_variables():
        print("Parameter:", v.name, "device:", v.device, "shape:", v.get_shape())
    print("Trainable variables are:")
    for v in tf.trainable_variables():
        print("Parameter:", v.name, "device:", v.device, "shape:", v.get_shape())

def save_model_for_dense_tensor(init_model_path, output_model_path, model_name, var, optimizer, output_slots=True):
    if isinstance(var, variables.PartitionedVariable):
        var = list(var)
    else:
        var = [var]

    slot_names = optimizer.get_slot_names()
    partition_count = len(var)
    save_ops = []
    restore_ops = []
    for p in xrange(partition_count):
        vars_to_save = [var[p]] + [optimizer.get_slot(var[p], name) for name in slot_names]
        if any([x == None for x in vars_to_save]):
            continue
        if not output_slots:
            vars_to_save = [var[p]]
        with ops.colocate_with(var[p]):
            save_op = raw_key_ops.variables_save(output_model_path, model_name, p, vars_to_save)
            save_ops.append(save_op)
            if init_model_path != "":
                restore_op = raw_key_ops.variables_restore(init_model_path, model_name, partition_count, p, vars_to_save)
                restore_ops.append(restore_op)
    return save_ops, restore_ops

def save_model_for_raw_key(init_model_path, output_model_path, model_name, hash_table, var, optimizer, output_slots=True):
    # get all slots for optimizer
    var_is_partitioned = isinstance(var, variables.PartitionedVariable)
    hash_table_is_partitioned = isinstance(hash_table, variables.PartitionedVariable)
    assert var_is_partitioned == hash_table_is_partitioned
    if var_is_partitioned:
        var = list(var)
        hash_table = list(hash_table)
        assert len(var) == len(hash_table)
    else:
        var = [var]
        hash_table = [hash_table]

    slot_names = optimizer.get_slot_names()
    partition_count = len(var)
    save_ops = []
    restore_ops = []
    for p in xrange(partition_count):
        vars_to_save = [var[p]] + [optimizer.get_slot(var[p], name) for name in slot_names]
        if any([x == None for x in vars_to_save]):
            continue
        if not output_slots:
            vars_to_save = [var[p]]
        with ops.colocate_with(var[p]):
            save_op = raw_key_ops.hash_variables_save(output_model_path, model_name, p, hash_table[p], vars_to_save)
            save_ops.append(save_op)
            if init_model_path != "":
                restore_op = raw_key_ops.hash_variables_restore(init_model_path, model_name, partition_count, p, hash_table[p], vars_to_save)
                restore_ops.append(restore_op)
    return save_ops, restore_ops
