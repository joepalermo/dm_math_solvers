from hparams import HParams
# hparams = HParams('.', hparams_filename='hparams', name='rl_math')
hparams = HParams('.', hparams_filename='hparams', name='rl_math', ask_before_deletion=False)
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from modelling.train_utils import init_trajectory_data_structures, init_envs, train, run_eval, fill_buffer, \
    get_logdir, StepDataset
from modelling.transformer_encoder import TransformerEncoderModel
import numpy as np
from sqlitedict import SqliteDict
from modelling.cache_utils import extract_trajectory_cache, visualize_trajectory_cache

# basic setup and checks
torch.manual_seed(hparams.run.seed)
np.random.seed(seed=hparams.run.seed)
device = torch.device(f'cuda:{hparams.run.gpu_id}' if torch.cuda.is_available() else 'cpu')
assert hparams.train.mode == 'positive_only' or hparams.train.mode == 'balanced'
writer = SummaryWriter(log_dir=get_logdir())

# initialize all environments
envs = init_envs(hparams.env)
rewarded_trajectories, trajectory_statistics = init_trajectory_data_structures(envs[0])

# load or init model
ntoken = hparams.env.vocab_size + 1
num_outputs = len(envs[0].actions)
if hparams.model.model_filepath is not None:
    model = torch.load(hparams.model.model_filepath)
else:
    model = TransformerEncoderModel(ntoken=ntoken, num_outputs=num_outputs, device=device)

# training loop
batch_i = 0
last_eval_batch_i = 0
# init replay buffer from trajectory cache on disk
replay_buffer = extract_trajectory_cache(hparams.env.trajectory_cache_filepath)
for buffer_i in tqdm(range(hparams.train.num_buffers)):
    model_to_use = None if len(replay_buffer) < hparams.train.min_saved_trajectories_until_training else model
    latest_buffer = fill_buffer(model_to_use, envs, trajectory_statistics, mode=hparams.train.mode)
    replay_buffer.extend(latest_buffer)
    if len(replay_buffer) > hparams.train.min_saved_trajectories_until_training:
        # construct dataset
        step_dataset = StepDataset(replay_buffer, model.device)
        # construct data loader
        data_loader = DataLoader(step_dataset, batch_size=model.batch_size, shuffle=True, drop_last=True)
        # train
        batches_in_dataset = len(step_dataset) // model.batch_size
        batches_to_train = min(batches_in_dataset, hparams.train.batches_per_train)
        batch_i = train(model, data_loader, batches_to_train, writer, batch_i)
        # eval
        if batch_i - last_eval_batch_i >= hparams.train.batches_per_eval:
            last_eval_batch_i = batch_i
            run_eval(model, envs, writer, batch_i, hparams.train.n_required_validation_episodes)
    trajectory_cache = SqliteDict(hparams.env.trajectory_cache_filepath, autocommit=True)
    visualize_trajectory_cache(envs[0].decode, trajectory_cache)

