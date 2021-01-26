from hparams import HParams
# hparams = HParams('.', hparams_filename='hparams', name='rl_math')
hparams = HParams('.', hparams_filename='hparams', name='rl_math', ask_before_deletion=False)
from tqdm import tqdm
import torch
from modelling.train_utils import init_trajectory_data_structures, init_envs, fill_buffer, visualize_trajectory_cache
import numpy as np
from sqlitedict import SqliteDict

torch.manual_seed(hparams.run.seed)
np.random.seed(seed=hparams.run.seed)

# basic setup and checks
assert hparams.train.mode == 'positive_only' or hparams.train.mode == 'balanced'

# initialize all environments
envs = init_envs(hparams.env)
rewarded_trajectories, rewarded_trajectory_statistics = init_trajectory_data_structures(envs[0])
for buffer_i in tqdm(range(hparams.train.num_buffers)):
    trajectory_buffer = fill_buffer(None, envs, hparams.train.buffer_threshold, hparams.train.positive_to_negative_ratio, rewarded_trajectories,
                         rewarded_trajectory_statistics, mode=hparams.train.mode, max_num_steps=hparams.train.fill_buffer_max_steps, verbose=False)

trajectory_cache = SqliteDict(hparams.env.trajectory_cache_filepath, autocommit=True)
visualize_trajectory_cache(envs[0].decode, trajectory_cache, num_to_sample=25)
trajectory_cache.close()
