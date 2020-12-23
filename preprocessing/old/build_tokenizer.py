import glob
from utils import recreate_dirpath
from preprocessing import build_tokenizer

# delete old tokenizer
tokenizers_dirpath = "output/tokenizers"
recreate_dirpath(tokenizers_dirpath)

# define inputs
input_file_pattern = "mathematics_dataset-v1.0/train*/*.txt"
input_filepaths = glob.glob(input_file_pattern)

# build tokenizer
build_tokenizer(input_filepaths, tokenizers_dirpath)
