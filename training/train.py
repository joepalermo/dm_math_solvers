import tensorflow as tf
import numpy as np
import os
import argparse
from utils import get_logger
from preprocessing import load_train, build_train_and_val_datasets
from training.params import TransformerParams
from training.transformer import Transformer

np.random.seed(1234)
tf.random.set_seed(1234)
# os.environ['CUDA_VISIBLE_DEVICES'] = "3"

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--eager', metavar='eager_mode', type=bool, default=True, help='Eager mode on, else Autograph')
parser.add_argument('--gpu_id', metavar='gpu_id', type=str, default="1", help='The selected GPU to use, default 1')
args = parser.parse_args()

tf.config.experimental_run_functions_eagerly(args.eager)
os.environ['CUDA_VISIBLE_DEVICES'] = "3"


if __name__ == '__main__':
    params = TransformerParams()
    logger = get_logger('validation', params.experiment_dir)
    logger.info("Logging to {}".format(params.experiment_dir))
    q_train, a_train = load_train('easy', num_files_to_include=1)
    train_ds, val_ds = build_train_and_val_datasets(q_train, a_train, TransformerParams())
    model = Transformer(params)
    model.train(params, train_ds, val_ds, logger)
