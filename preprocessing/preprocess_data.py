import glob
from preprocessing import load_tokenizers, preprocess_data
from utils import recreate_dirpath

# delete old preprocessed data
train_easy_preprocessed_dirpath = 'output/train_easy_preprocessed'
train_medium_preprocessed_dirpath = 'output/train_medium_preprocessed'
train_hard_preprocessed_dirpath = 'output/train_hard_preprocessed'
recreate_dirpath(train_easy_preprocessed_dirpath)
recreate_dirpath(train_medium_preprocessed_dirpath)
recreate_dirpath(train_hard_preprocessed_dirpath)

# delete old preprocessed test data
test_interpolate_preprocessed_dirpath = 'output/test_interpolate_preprocessed'
test_extrapolate_preprocessed_dirpath = 'output/test_extrapolate_preprocessed'
recreate_dirpath(test_interpolate_preprocessed_dirpath)
recreate_dirpath(test_extrapolate_preprocessed_dirpath)

# define training data inputs
train_easy_file_pattern = 'mathematics_dataset-v1.0/train-easy/*.txt'
train_medium_file_pattern = 'mathematics_dataset-v1.0/train-medium/*.txt'
train_hard_file_pattern = 'mathematics_dataset-v1.0/train-hard/*.txt'
train_easy_filepaths = glob.glob(train_easy_file_pattern)
train_medium_filepaths = glob.glob(train_medium_file_pattern)
train_hard_filepaths = glob.glob(train_hard_file_pattern)

# define test data inputs
interpolate_file_pattern = 'mathematics_dataset-v1.0/interpolate/*.txt'
extrapolate_file_pattern = 'mathematics_dataset-v1.0/extrapolate/*.txt'
interpolate_filepaths = glob.glob(interpolate_file_pattern)
extrapolate_filepaths = glob.glob(extrapolate_file_pattern)

# load tokenizer
tokenizer_dirpath = 'output/tokenizers'
char2idx, idx2char = load_tokenizers(tokenizer_dirpath)

# preprocess
preprocess_data(train_easy_filepaths, char2idx, output_dirpath=train_easy_preprocessed_dirpath)
preprocess_data(train_medium_filepaths, char2idx, output_dirpath=train_medium_preprocessed_dirpath)
preprocess_data(train_hard_filepaths, char2idx, output_dirpath=train_hard_preprocessed_dirpath)
preprocess_data(interpolate_filepaths, char2idx, output_dirpath=test_interpolate_preprocessed_dirpath)
preprocess_data(extrapolate_filepaths, char2idx, output_dirpath=test_extrapolate_preprocessed_dirpath)


