import glob
import numpy as np
from utils import recreate_dirpath, read_text_file
from preprocessing import load_tokenizers


def decode(tokens, idx2char):
    # filter out special tokens
    tokens = [idx for idx in tokens if idx not in [0, 1, 2]]
    # decode
    tokens = [idx2char[idx] for idx in tokens]
    return tokens


def matching_filepath(preprocessed_fp, raw_filepaths):
    preprocessed_identifier = '_'.join(preprocessed_fp.split('/')[-1].split('_')[:-1])
    print(preprocessed_identifier)
    for raw_fp in raw_filepaths:
        if preprocessed_identifier in raw_fp:
            return raw_fp
    raise Exception("missing raw filepath")


# TODO: use pytest
def decode_and_match_first(paired_filepaths, idx2char):
    recreate_dirpath('output/decoded_preprocessed')
    for preprocessed_fp, raw_fp in paired_filepaths:
        # get the first set of encoded tokens
        tokens = np.load(preprocessed_fp)[0].tolist()
        decoded_tokens = decode(tokens, idx2char)
        decoded_line = "".join(decoded_tokens)
        # get the corresponding raw tokens
        if 'question' in preprocessed_fp:
            tokens = read_text_file(raw_fp).split('\n')[0]
        elif 'answer' in preprocessed_fp:
            tokens = read_text_file(raw_fp).split('\n')[1]
        else:
            raise Exception("must be Q or A")
        line = "".join(tokens)
        assert decoded_line == line, f"decoded line: {decoded_line} != line: {line}"


# get preprocessed filepaths
preprocessed_file_pattern = 'output/preprocessed/*'
preprocessed_filepaths = glob.glob(preprocessed_file_pattern)

# get corresponding raw filepaths
raw_filepaths = glob.glob('mathematics_dataset-v1.0/train*/*.txt')
paired_filepaths = [(preprocessed_fp, matching_filepath(preprocessed_fp, raw_filepaths))
                    for preprocessed_fp in preprocessed_filepaths]

# load tokenizers
tokenizer_dirpath = 'output/tokenizers'
char2idx, idx2char = load_tokenizers(tokenizer_dirpath)

# test
decode_and_match_first(paired_filepaths, idx2char)
