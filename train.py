import tensorflow as tf
import numpy as np
import os
import argparse
from utils import get_logger
from preprocessing import load_train, build_train_and_val_datasets
from transformer.params import TransformerParams
from transformer.transformer import Transformer

np.random.seed(1234)
tf.random.set_seed(1234)
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--eager', metavar='eager_mode', type=bool, default=True, help='Eager mode on, else Autograph')
parser.add_argument('--gpu_id', metavar='gpu_id', type=str, default="1", help='The selected GPU to use, default 1')
args = parser.parse_args()
tf.config.experimental_run_functions_eagerly(args.eager)
params = TransformerParams()
os.environ['CUDA_VISIBLE_DEVICES'] = params.gpu_id

if __name__ == '__main__':
    logger = get_logger('validation', params.experiment_dirpath)
    logger.info("Logging to {}".format(params.experiment_dirpath))
    module_name_to_arrays = load_train('easy', num_files_to_include=10)
    train_ds, module_name_to_val_ds = build_train_and_val_datasets(module_name_to_arrays, TransformerParams())
    model = Transformer(params)
    model.load_latest_checkpoint()
    model.train(params, train_ds, module_name_to_val_ds, logger)