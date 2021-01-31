from hparams import HParams
hparams = HParams('.', hparams_filename='hparams', name='rl_math', ask_before_deletion=False)
from modelling.train_utils import init_envs
from modelling.cache_utils import extract_trajectory_cache, visualize_trajectory_cache
from sqlitedict import SqliteDict

envs = init_envs(hparams.env)
trajectory_cache = SqliteDict(hparams.env.trajectory_cache_filepath, autocommit=True)
visualize_trajectory_cache(envs[0].decode, trajectory_cache, num_to_sample=25)
trajectory_cache.close()
