from hparams import HParams
hparams = HParams('.', hparams_filename='hparams', name='rl_math', ask_before_deletion=False)
from modelling.train_utils import extract_all_steps_from_trajectory_cache

steps = extract_all_steps_from_trajectory_cache(hparams.env.trajectory_cache_filepath, verbose=True)

