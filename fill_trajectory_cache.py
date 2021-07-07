from hparams import HParams
# hparams = HParams('.', hparams_filename='hparams', name='rl_math')
hparams = HParams('.', hparams_filename='hparams', name='rl_math', ask_before_deletion=False)
from tqdm import tqdm
from modelling.train_utils import init_trajectory_data_structures, init_envs, fill_buffer
from modelling.cache_utils import visualize_trajectory_cache
from sqlitedict import SqliteDict

assert hparams.train.fill_buffer_mode != 'anything'

# initialize all environments
envs = init_envs(hparams.env)
trajectory_statistics = init_trajectory_data_structures(envs[0])
for buffer_i in tqdm(range(hparams.train.num_epochs)):
    trajectory_buffer = fill_buffer(None, envs, trajectory_statistics,
                                    hparams.train.random_exploration_trajectory_cache_filepath)

trajectory_cache = SqliteDict(hparams.train.random_exploration_trajectory_cache_filepath, autocommit=True)
visualize_trajectory_cache(envs[0].decode, trajectory_cache, num_to_sample=25)
trajectory_cache.close()

