import glob
from preprocessing import load_tokenizers, preprocess_data
from utils import recreate_dirpath

# delete old preprocessed data
preprocessed_dirpath = 'output/preprocessed'
recreate_dirpath(preprocessed_dirpath)

# load tokenizer
tokenizer_dirpath = 'output/tokenizers'
char2idx, idx2char = load_tokenizers(tokenizer_dirpath)

# define inputs
input_file_pattern = 'mathematics_dataset-v1.0/train-easy/*.txt'
input_filepaths = glob.glob(input_file_pattern)

# preprocess data
preprocess_data(input_filepaths, char2idx, output_dirpath=preprocessed_dirpath)

