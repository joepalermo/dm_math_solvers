import tensorflow as tf
import numpy as np
import os
import argparse
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
    model = Transformer(params)
    model.load_latest_checkpoint()
    output = model.raw_inference("what is 2 + 2?")
    print(output)