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
import scipy
from utils import flatten
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
trajectory_statistics = init_trajectory_data_structures(envs[0])

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
replay_buffer = np.array(flatten(extract_trajectory_cache(hparams.env.trajectory_cache_filepath)))
replay_priority = np.ones(len(replay_buffer)) * hparams.train.max_td_error
for buffer_i in range(hparams.train.num_buffers):
    # model_to_use = None if len(replay_buffer) < hparams.train.min_saved_trajectories_until_training else model
    # model_to_use = None
    # latest_buffer = fill_buffer(model_to_use, envs, trajectory_statistics)
    # replay_buffer.extend(latest_buffer)
    if len(replay_buffer) > hparams.train.min_saved_trajectories_until_training:
        # sample from prioritized replay buffer
        replay_probability = scipy.special.softmax(replay_priority)
        steps_to_sample = hparams.train.batch_size * hparams.train.batches_per_train
        sampled_idxs = np.random.choice(np.arange(len(replay_buffer)), size=steps_to_sample, p=replay_probability)
        sampled_steps = replay_buffer[sampled_idxs]
        # construct data loader
        step_dataset = StepDataset(sampled_steps, model.device)
        data_loader = DataLoader(step_dataset, batch_size=model.batch_size, shuffle=False, drop_last=True)
        # train
        batch_i, td_error = train(model, data_loader, writer, batch_i)
        replay_priority[sampled_idxs] = td_error.cpu().detach().numpy() ** hparams.train.prioritization_exponent
        idx_to_print = replay_priority.argsort()[-5:]
        for x in replay_buffer[idx_to_print]:
            print(envs[0].decode(x[0]))
        print(batch_i)
        # eval
        if batch_i - last_eval_batch_i >= hparams.train.batches_per_eval:
            last_eval_batch_i = batch_i
            run_eval(model, envs, writer, batch_i, hparams.train.n_required_validation_episodes)
    # trajectory_cache = SqliteDict(hparams.env.trajectory_cache_filepath, autocommit=True)
    # visualize_trajectory_cache(envs[0].decode, trajectory_cache)

