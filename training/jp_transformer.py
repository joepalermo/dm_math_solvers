import os
import glob
import numpy as np
import tensorflow as tf


# define inputs
input_file_pattern = 'output/preprocessed/*.npy'
input_filepaths = iter(sorted(glob.glob(input_file_pattern)))

all_q = list()
all_a = list()
for a_filepath in input_filepaths:
    q_filepath = next(input_filepaths)
    q = np.load(q_filepath)
    a = np.load(a_filepath)
    print(q.shape, a.shape)
    all_q.append(q)
    all_a.append(a)
q = np.concatenate(all_q, axis=0)
a = np.concatenate(all_a, axis=0)
print(q.shape, a.shape)

ds = tf.data.Dataset.from_tensor_slices((q,a))
ds = ds.batch(64)

# ds = tf.data.TFRecordDataset("output/tf_record/test.tfrecords", compression_type=None, buffer_size=None, num_parallel_reads=None)

for (batch, (inp, tar)) in enumerate(ds):
    print(inp.shape, tar.shape)
