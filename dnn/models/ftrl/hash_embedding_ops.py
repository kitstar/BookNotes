"""Operations for woodblocks raw key embeddings."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.user_ops import user_ops
import tensorflow as tf

#woodblocks raw key ops
import raw_key_ops

def _do_hash_gather(hash_table, params, keys, poisson=0.1, name=None):
  """Hash gather with possin inclusios
  """
  ids = raw_key_ops.hash_to_id(hash_table, keys, poisson)
  filtered_ids, idx = raw_key_ops.filter(ids, -1)

  # add hashing table to variable list
  values = array_ops.gather(params, filtered_ids, name=name)
  return values, idx

def hash_embedding_lookup(hash_table, params, keys, poisson=0.1, partition_strategy="mod", name=None):
  """Looks up raw woodblocks features in a list of embedding tensors.
  """
  if params in (None, (), []):
    raise ValueError("Need at least one param")
  if isinstance(params, variables.PartitionedVariable):
    params = list(params)  # Iterate to get the underlying Variables.
  if not isinstance(params, list):
    params = [params]

  if isinstance(hash_table, variables.PartitionedVariable):
    hash_table = list(hash_table)
  if not isinstance(hash_table, list):
    hash_table = [hash_table]

  with ops.name_scope(name, "hash_embedding_lookup", hash_table + params + [keys]) as name:
    #print("name scope name:", name)
    np = len(params)  # Number of partitions
    # Preserve the resource variable status to avoid accidental dense reads.
    if not any(isinstance(p, resource_variable_ops.ResourceVariable)
               for p in params):
      params = ops.convert_n_to_tensor_or_indexed_slices(params, name="params")
    if np == 1:
      with ops.colocate_with(params[0]):
        values, idx = _do_hash_gather(hash_table[0], params[0], keys, poisson=poisson, name=name)
        #padding non-existing features
        return tf.scatter_nd(tf.reshape(idx, [-1, 1]), values, raw_key_ops.modify_first_dim(tf.shape(keys), tf.shape(values)))
    else:
      original_indices = math_ops.range(array_ops.shape(keys, out_type=tf.int32)[0])
      # murmur hash32 for partition
      key_hashes = raw_key_ops.murmur32(keys)
      if partition_strategy == "mod":
        p_assignments = key_hashes % np
      else:
        raise ValueError("Unrecognized partition strategy: " +
                         partition_strategy)

      # Cast partition assignments to int32 for use in dynamic_partition.
      # There really should not be more than 2^32 partitions.
      p_assignments = math_ops.cast(p_assignments, dtypes.int32)

      #gather_keys = data_flow_ops.dynamic_partition(keys, p_assignments, np)
      ## Similarly, partition the original indices.
      #pindices = data_flow_ops.dynamic_partition(original_indices,
      #                                           p_assignments, np)
      gather_keys = raw_key_ops.fast_dynamic_partition(keys, p_assignments, np)
      pindices = raw_key_ops.fast_dynamic_partition(original_indices, p_assignments, np)

      # Do np separate lookups, finding embeddings for plist[p] in params[p]
      partitioned_result_partial = []
      for p in xrange(np):
        with ops.colocate_with(params[p]):
          partitioned_result_partial.append(_do_hash_gather(hash_table[p], params[p], gather_keys[p], poisson=poisson))

      # Stitch these back together
      # padding zero for non-existing features
      partitioned_result = []
      for p in xrange(np):
        partitioned_result.append(tf.scatter_nd(tf.reshape(partitioned_result_partial[p][1], [-1, 1]),
                                                partitioned_result_partial[p][0],
                                                raw_key_ops.modify_first_dim(tf.shape(gather_keys[p]), tf.shape(partitioned_result_partial[p][0]))
                                                ))

      # filter non-existing features
      ret = data_flow_ops.dynamic_stitch(pindices, partitioned_result,
                                         name=name)
      element_shape = params[0].get_shape()[1:]
      return array_ops.reshape(ret, array_ops.concat([array_ops.shape(key_hashes), element_shape], 0))

def hash_embedding_lookup_sparse(hash_table, params, sp_indices, sp_values, sp_shape, poisson=0.1, name=None, combiner="sum", max_norm=None):
  """
  woodblocks raw key embedding lookup sparse
  woodblocks raw key is represented as int32[3]
  """
  if combiner not in ("mean", "sqrtn", "sum"):
    raise ValueError("combiner must be one of 'mean', 'sqrtn' or 'sum'")
  if isinstance(hash_table, variables.PartitionedVariable):
    hash_table = list(hash_table)
  if not isinstance(hash_table, list):
    hash_table = [hash_table]

  if isinstance(params, variables.PartitionedVariable):
    params = list(params)  # Iterate to get the underlying Variables.
  if not isinstance(params, list):
    params = [params]

  with ops.name_scope(name, "hash_embedding_lookup_sparse",
                      params + hash_table + [sp_indices, sp_values, sp_shape]) as name:
    segment_ids = sp_indices[:, 0]
    if segment_ids.dtype != dtypes.int32:
      segment_ids = math_ops.cast(segment_ids, dtypes.int32)

    raw_keys = sp_values
    raw_keys, idx = raw_key_ops.woodblocks_unique(raw_keys)

    embeddings = hash_embedding_lookup(hash_table, params, raw_keys, poisson=poisson)

    assert idx is not None
    if combiner == "sum":
      embeddings = math_ops.sparse_segment_sum(embeddings, idx, segment_ids,
                                               name=name)
    elif combiner == "mean":
      embeddings = math_ops.sparse_segment_mean(embeddings, idx, segment_ids,
                                                name=name)
    elif combiner == "sqrtn":
      embeddings = math_ops.sparse_segment_sqrt_n(embeddings, idx,
                                                  segment_ids, name=name)
    else:
      assert False, "Unrecognized combiner"

    return embeddings
