import tensorflow as tf

def split_into_intervals(total_length, interval_length):
    """
    Return a bunch of intervals at a certain stride length. They
    will all intersect a bit.
    Useful for going through data that doesn't fit in the transformer
    memory.
    :param total_length: The total length that the intervals should span
    :param interval_length: The length of any given interval.
    :return:
    """
    if total_length <= interval_length:
        yield 0, total_length
        return
    effective_length = total_length - interval_length
    goal_stride = interval_length * 3 // 4  # Consecutive intervals share about 1/4th of their data
    n_intervals = max(effective_length // goal_stride, 1)
    for i in range(n_intervals + 1):
        st = (effective_length * i) // n_intervals
        yield st, st + interval_length


def get_from_map_subrange(map, key, start, end, offset=False):
  el = map.get(key)
  if el is not None:
    el = el[:, start:end]
    if offset:
      el -= start
  return el

def get_row_lengths(inds):
  return tf.math.reduce_max(inds, axis=1, name='get_row_lengths') + 1

def init_entity_matrix(dim1, dim2, dim3):
  """
  Make ragged tensor of shape (dim1, (dim2), dim3)
  with all values initalized to 0
  """
  matricies = [
               tf.Variable(tf.zeros((dim2[i], dim3)))
               for i in range(dim1)
  ]
  return matricies

def dict_subset(input_data, start, end):
    input_data_in_range = {}
    for key in ['input_ids', 'attention_mask', 'entity_ends', 'to_embed_ind']:
        if key in input_data:
            input_data_in_range[key] = input_data[key][:, start:end]
            if key in ['entity_ends', 'to_embed_ind']:
                input_data_in_range[key] = input_data[key] - start
    return input_data_in_range

