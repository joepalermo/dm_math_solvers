from hparams import HParams
hparams = HParams('.', hparams_filename='hparams', name='rl_math')
import os
from tqdm import tqdm

problem_filepaths = [os.path.join(hparams.env.all_data_dirpath, filename) for filename in hparams.env.selected_filenames]
train_problem_filepaths = [os.path.join(hparams.env.data_dirpath, filename) for filename in hparams.env.selected_filenames]
test_problem_filepaths = [os.path.join(hparams.env.test_data_dirpath, filename) for filename in hparams.env.selected_filenames]

if os.path.isdir(hparams.env.data_dirpath) or os.path.isdir(hparams.env.test_data_dirpath):
    raise ValueError(f"data directories already exist")
else:
    os.mkdir(hparams.env.data_dirpath)
    os.mkdir(hparams.env.test_data_dirpath)


for filepath, train_filepath, test_filepath in tqdm(zip(problem_filepaths, train_problem_filepaths, test_problem_filepaths)):
    #read data
    with open(filepath, "r") as f:
        lines = f.readlines()
    num_pairs = len(lines) // 2
    num_train_pairs = int((1 - hparams.env.test_percentage) * num_pairs)

    # Write data
    with open(train_filepath, "w") as f:
        f.writelines(lines[:2 * num_train_pairs])
    with open(test_filepath, "w") as f:
        f.writelines(lines[2 * num_train_pairs:])
print("train and test datasets have been created")