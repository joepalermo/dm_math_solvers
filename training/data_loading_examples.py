from training.params import TransformerParams
from preprocessing import load_train, build_train_and_val_datasets





q_train, a_train = load_train('easy', num_files_to_include=2)
train_ds, val_ds = build_train_and_val_datasets(q_train, a_train, TransformerParams())
for (batch, (inp, tar)) in enumerate(train_ds):
    print(inp.shape, tar.shape)

# ds = tf.data.TFRecordDataset("output/tf_record/test.tfrecords", compression_type=None, buffer_size=None, num_parallel_reads=None)

