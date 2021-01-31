from hparams import HParams
hparams = HParams('.', hparams_filename='hparams', name='rl_math', ask_before_deletion=False)
from modelling.cache_utils import extract_trajectory_cache

trajectories = extract_trajectory_cache(hparams.env.trajectory_cache_filepath, verbose=True)


