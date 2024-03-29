from hparams import HParams
hparams = HParams('.', hparams_filename='hparams', name='rl_math', ask_before_deletion=False)
from modelling.train_utils import init_envs
from modelling.cache_utils import extract_trajectory_cache, visualize_trajectory_cache, \
    visualize_trajectory_cache_by_module_and_difficulty
from sqlitedict import SqliteDict

envs = init_envs(hparams.env)
trajectory_cache = SqliteDict(hparams.train.random_exploration_trajectory_cache_filepath, autocommit=True)
visualize_trajectory_cache_by_module_and_difficulty(envs[0].decode, trajectory_cache,
                                                    question_length=hparams.env.max_sequence_length, num_to_sample=5)
# extract_trajectory_cache(hparams.train.random_exploration_trajectory_cache_filepath, verbose=True)
trajectory_cache.close()
