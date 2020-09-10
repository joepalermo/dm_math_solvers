import glob
import numpy as np
from utils import recreate_dirpath
from preprocessing import load_tokenizers

# TODO: use pytest
def decode_and_match_first(input_filepaths, idx2char):
    recreate_dirpath('output/decoded_preprocessed')
    for fp in input_filepaths:
        fn = fp.split('/')[-1]
        # print(f"decoding {fn}...")
        line = np.load(fp)[0].tolist()
        # filter out special tokens
        line = [idx for idx in line if idx not in [0, 1, 2]]
        # decode
        line = [idx2char[idx] for idx in line]
        # TODO: replace with assert
        print(line)

# define inputs
# input_file_pattern = 'mathematics_dataset-v1.0/train-easy/algebra__linear_1d.txt'
input_file_pattern = 'output/preprocessed/*'
input_filepaths = glob.glob(input_file_pattern)

# load tokenizers
tokenizer_dirpath = 'output/tokenizers'
char2idx, idx2char = load_tokenizers(tokenizer_dirpath)

decode_and_match_first(input_filepaths, idx2char)
