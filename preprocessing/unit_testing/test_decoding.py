import glob
import numpy as np
from utils import recreate_dirpath, read_text_file
from preprocessing import load_tokenizers, load_train, build_train_and_val_datasets
from transformer.params import TransformerParams


def decode(tokens, idx2char):
    # filter out special tokens
    tokens = [idx for idx in tokens if idx not in [0, 1, 2]]
    # decode
    tokens = [idx2char[idx] for idx in tokens]
    return tokens


def matching_filepath(preprocessed_fp, raw_filepaths):
    preprocessed_identifier = '_'.join(preprocessed_fp.split('/')[-1].split('_')[:-1])
    for raw_fp in raw_filepaths:
        raw_fn = raw_fp.split('/')[-1]
        if preprocessed_identifier + '.txt' == raw_fn:
            return raw_fp
    raise Exception(f"missing filepath: {preprocessed_fp}")


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
        print(decoded_line, '==?', line)
        assert decoded_line == line, f"decoded line: {decoded_line} != line: {line}, from {preprocessed_fp} and {raw_fp}"


def inspect_batch(input_batch, target_batch, idx2char, num_to_inspect=1):
    for i in range(num_to_inspect):
        first_inp = input_batch[i: i + 1]
        first_target = target_batch[i]
        print("question: ", "".join(decode(first_inp[0], idx2char)))
        print("answer  : ", "".join(decode(first_target, idx2char)))


# get preprocessed filepaths
preprocessed_file_pattern = 'output/train_easy_preprocessed/*'
preprocessed_filepaths = glob.glob(preprocessed_file_pattern)

# get corresponding raw filepaths
raw_filepaths = glob.glob('mathematics_dataset-v1.0/train-easy/*.txt')
paired_filepaths = [(preprocessed_fp, matching_filepath(preprocessed_fp, raw_filepaths))
                    for preprocessed_fp in preprocessed_filepaths]

# load tokenizers
tokenizer_dirpath = 'output/tokenizers'
char2idx, idx2char = load_tokenizers(tokenizer_dirpath)

# # test
# decode_and_match_first(paired_filepaths, idx2char)

# test tf dataset
q_train, a_train = load_train('easy', num_files_to_include=2)
train_ds, val_ds = build_train_and_val_datasets(q_train, a_train, TransformerParams())
for batch, (input_batch, target_batch) in enumerate(train_ds):
    input_batch = input_batch.numpy()
    target_batch = target_batch.numpy()
    inspect_batch(input_batch, target_batch, idx2char, num_to_inspect=15)
    break