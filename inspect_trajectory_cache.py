from hparams import HParams
# hparams = HParams('.', hparams_filename='hparams', name='rl_math')
hparams = HParams('.', hparams_filename='hparams', name='rl_math', ask_before_deletion=False)
from tqdm import tqdm
import torch
from modelling.train_utils import extract_all_steps_from_trajectory_cache
import numpy as np
from sqlitedict import SqliteDict

torch.manual_seed(hparams.run.seed)
np.random.seed(seed=hparams.run.seed)

# basic setup and checks
assert hparams.train.mode == 'positive_only' or hparams.train.mode == 'balanced'

steps = extract_all_steps_from_trajectory_cache(hparams.env.trajectory_cache_filepath)

