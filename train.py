from hparams import HParams
hparams = HParams('.', hparams_filename='hparams', name='rl_math')
# hparams = HParams('.', hparams_filename='hparams', name='rl_math', ask_before_deletion=False)
import torch
import random
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from modelling.cache_utils import extract_replay_buffer_from_trajectory_cache
from modelling.train_utils import init_trajectory_data_structures, init_envs, train, run_eval, get_logdir, StepDataset,\
    fill_buffer, get_td_error
from modelling.transformer_encoder import TransformerEncoderModel
import numpy as np
from utils import flatten
import os
from modelling.cache_utils import extract_trajectory_cache

# basic setup and checks
torch.manual_seed(hparams.run.seed)
np.random.seed(seed=hparams.run.seed)
device = torch.device(f'cuda:{hparams.run.gpu_id}' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(log_dir=get_logdir())

# initialize all environments
envs = init_envs(hparams.env)
trajectory_statistics = init_trajectory_data_structures(envs[0])

# init models
ntoken = hparams.env.vocab_size + 1
num_outputs = len(envs[0].actions)
max_num_nodes = envs[0].max_num_nodes
network = TransformerEncoderModel(ntoken=ntoken, num_outputs=num_outputs, max_num_nodes=max_num_nodes, device=device)
target_network = TransformerEncoderModel(ntoken=ntoken, num_outputs=num_outputs, max_num_nodes=max_num_nodes,
                                         device=device)
target_network.eval()

# init replay buffer from trajectory cache on disk
replay_buffer = extract_replay_buffer_from_trajectory_cache(hparams.train.random_exploration_trajectory_cache_filepath,
                                                            hparams.train.replay_buffer_size)
replay_priority = np.ones(len(replay_buffer)) * hparams.train.default_replay_buffer_priority

# training loop --------------------------------------------------------------------------------------------------------


def extract_strings_from_batches(batches, env):
    strings = []
    for batch in batches:
        state_batch, action_batch = batch
        for state, action in zip(state_batch, action_batch):
            decoded_state = env.decode(state)
            strings.append(f'{decoded_state}, action: {action}')
    return "\n".join(strings)


def log_to_text_file(string):
    filepath = os.path.join(get_logdir(), hparams.run.logging_text_filename)
    if os.path.isfile(filepath):
        mode = 'a'
    else:
        mode = 'w'
    with open(filepath, mode) as f:
        f.write(string + '\n')


added_to_replay_buffer = 0
batch_i = last_eval_batch_i = last_target_network_update_batch_i = 0
# assert that one of these is true, otherwise nothing will happen
assert (batch_i >= hparams.train.num_batches_until_fill_buffer) or \
       (len(replay_buffer) >= hparams.train.min_saved_steps_until_training)

for epoch_i in range(hparams.train.num_epochs):
    print(f'epoch #{epoch_i}')

    # fill buffer -----------
    print(f'fresh replay buffer: {round(added_to_replay_buffer / len(replay_buffer) * 100, 2)}%')
    if batch_i >= hparams.train.num_batches_until_fill_buffer:
        latest_buffer = fill_buffer(network, envs, trajectory_statistics,
                                    hparams.train.model_exploration_trajectory_cache_filepath)
        latest_buffer = np.array(flatten(latest_buffer))
        latest_replay_priority = np.ones(len(latest_buffer)) * hparams.train.default_replay_buffer_priority
        # replace oldest
        replay_buffer = np.concatenate([replay_buffer[len(latest_buffer):], latest_buffer])
        replay_priority = np.concatenate([replay_priority[len(latest_replay_priority):], latest_replay_priority])
        # # replace lowest priority
        # lowest_priority_indices = np.argsort(replay_priority)[:len(latest_buffer)]
        # replay_buffer[lowest_priority_indices] = latest_buffer
        # replay_priority[lowest_priority_indices] = latest_replay_priority
        assert len(replay_buffer) == hparams.train.replay_buffer_size and \
               len(replay_priority) == hparams.train.replay_buffer_size
        added_to_replay_buffer += len(latest_buffer)

    # train -----------

    if len(replay_buffer) > hparams.train.min_saved_steps_until_training:

        # sample from prioritized replay buffer
        replay_probability = replay_priority / np.sum(replay_priority)
        steps_to_sample = hparams.train.batch_size * hparams.train.batches_per_epoch
        if batch_i < hparams.train.batches_until_stop_replay_priority:
            sampled_idxs = np.random.choice(np.arange(len(replay_buffer)), size=steps_to_sample, p=replay_probability)
        else:
            sampled_idxs = np.random.choice(np.arange(len(replay_buffer)), size=steps_to_sample)
        sampled_steps = replay_buffer[sampled_idxs]

        # construct data loader
        step_dataset = StepDataset(sampled_steps, network.device)
        data_loader = DataLoader(step_dataset, batch_size=network.batch_size, shuffle=False, drop_last=True)

        # train
        print(f'batch #{batch_i}')
        if hparams.train.use_target_network:
            batch_i, _, batches = train(network, target_network, data_loader, writer, batch_i)
        else:
            batch_i, _, batches = train(network, None, data_loader, writer, batch_i)
        # logging
        batch_string = extract_strings_from_batches(batches, envs[0])
        log_to_text_file(f'batch #{batch_i}')
        log_to_text_file(batch_string)

        #Sample indices for computing td error
        sampled_idxs = np.random.choice(np.arange(len(replay_buffer)),
                                        size=hparams.train.n_batch_td_error*hparams.train.sample_td_error_batch_size)

        sampled_steps = replay_buffer[sampled_idxs]
        td_error = get_td_error(network, sampled_steps)

        # update replay priority
        replay_priority[sampled_idxs] = td_error.cpu().detach().numpy() ** hparams.train.prioritization_exponent
        # visualize_replay_priority(envs, replay_priority, replay_buffer)

        # update target network
        if batch_i >= hparams.train.batches_until_target_network and \
                batch_i - last_target_network_update_batch_i >= hparams.train.batches_per_target_network_update:
            last_target_network_update_batch_i = batch_i
            target_network.load_state_dict(network.state_dict())
            target_network.eval()
            print('updated target network')

        # eval
        if batch_i - last_eval_batch_i >= hparams.train.batches_per_eval:
            last_eval_batch_i = batch_i
            mean_val_reward = run_eval(network, envs, writer, batch_i, hparams.train.n_required_validation_episodes)
            log_to_text_file(f'mean val reward: {mean_val_reward}')

