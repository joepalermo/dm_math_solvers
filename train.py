from hparams import HParams
# hparams = HParams('.', hparams_filename='hparams', name='rl_math')
hparams = HParams('.', hparams_filename='hparams', name='rl_math', ask_before_deletion=False)
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from modelling.train_utils import init_trajectory_data_structures, init_envs, train, run_eval, get_logdir, StepDataset
from modelling.transformer_encoder import TransformerEncoderModel
import numpy as np
from utils import flatten
from modelling.cache_utils import extract_trajectory_cache

# basic setup and checks
torch.manual_seed(hparams.run.seed)
np.random.seed(seed=hparams.run.seed)
device = torch.device(f'cuda:{hparams.run.gpu_id}' if torch.cuda.is_available() else 'cpu')
assert hparams.train.mode == 'positive_only' or hparams.train.mode == 'balanced'
writer = SummaryWriter(log_dir=get_logdir())

# initialize all environments
envs = init_envs(hparams.env)
trajectory_statistics = init_trajectory_data_structures(envs[0])

# init model
ntoken = hparams.env.vocab_size + 1
num_outputs = len(envs[0].actions)
network = TransformerEncoderModel(ntoken=ntoken, num_outputs=num_outputs, device=device)
target_network = TransformerEncoderModel(ntoken=ntoken, num_outputs=num_outputs, device=device)
target_network.eval()

# training loop
batch_i = last_eval_batch_i = last_target_network_update_batch_i = 0
# init replay buffer from trajectory cache on disk
# TODO: remove clip to 5k steps
replay_buffer = np.array(flatten(extract_trajectory_cache(hparams.env.trajectory_cache_filepath)))
replay_priority = np.ones(len(replay_buffer)) * hparams.train.default_replay_buffer_priority
for buffer_i in range(hparams.train.num_buffers):
    # model_to_use = None if len(replay_buffer) < hparams.train.min_saved_trajectories_until_training else network
    # model_to_use = None
    # latest_buffer = fill_buffer(model_to_use, envs, trajectory_statistics)
    # replay_buffer.extend(latest_buffer)
    if len(replay_buffer) > hparams.train.min_saved_trajectories_until_training:
        # sample from prioritized replay buffer
        replay_probability = replay_priority / np.sum(replay_priority)
        steps_to_sample = hparams.train.batch_size * hparams.train.batches_per_train
        sampled_idxs = np.random.choice(np.arange(len(replay_buffer)), size=steps_to_sample, p=replay_probability)
        sampled_steps = replay_buffer[sampled_idxs]
        # construct data loader
        step_dataset = StepDataset(sampled_steps, network.device)
        data_loader = DataLoader(step_dataset, batch_size=network.batch_size, shuffle=False, drop_last=True)
        # train
        if batch_i > 2000:
            batch_i, td_error = train(network, target_network, data_loader, writer, batch_i)
        else:
            batch_i, td_error = train(network, None, data_loader, writer, batch_i)
        replay_priority[sampled_idxs] = td_error.cpu().detach().numpy() ** hparams.train.prioritization_exponent
        if np.sum(replay_priority != 100) == len(replay_priority):
            pass
            # visualize_replay_priority(envs, replay_priority, replay_buffer)
        else:
            print(f"{np.sum(replay_priority != 100) / len(replay_priority) * 100} %")
        print(f'batch #{batch_i}')
        # update target network
        if batch_i - last_target_network_update_batch_i >= hparams.train.batches_per_target_network_update:
            last_target_network_update_batch_i = batch_i
            target_network.load_state_dict(network.state_dict())
            target_network.eval()
            print('updated target network')
        # eval
        if batch_i - last_eval_batch_i >= hparams.train.batches_per_eval:
            last_eval_batch_i = batch_i
            run_eval(network, envs, writer, batch_i, hparams.train.n_required_validation_episodes)
    # trajectory_cache = SqliteDict(hparams.env.trajectory_cache_filepath, autocommit=True)
    # visualize_trajectory_cache(envs[0].decode, trajectory_cache)

