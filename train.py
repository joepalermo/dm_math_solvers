from hparams import HParams
# hparams = HParams('.', hparams_filename='hparams', name='rl_math')
hparams = HParams('.', hparams_filename='hparams', name='rl_math', ask_before_deletion=False)
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from modelling.cache_utils import extract_replay_buffer_from_trajectory_cache, log_batches, \
    log_to_text_file, add_trajectory_return_to_trajectories
from modelling.train_utils import init_trajectory_data_structures, init_envs, train, run_eval, get_logdir, StepDataset,\
    fill_buffer, get_td_error
from modelling.transformer_encoder import TransformerEncoderModel
import numpy as np
import random
from utils import flatten
import os

# basic setup and checks
torch.manual_seed(hparams.run.seed)
np.random.seed(seed=hparams.run.seed)
device = torch.device(f'cuda:{hparams.run.gpu_id}' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(log_dir=get_logdir())
logging_batches_filepath = os.path.join(get_logdir(), hparams.run.logging_batches_filename)
logging_q_values_filepath = os.path.join(get_logdir(), hparams.run.logging_q_values_filename)

# initialize all environments
envs = init_envs(hparams.env)
trajectory_statistics = init_trajectory_data_structures(envs[0])

# init models
num_question_tokens = hparams.env.question_vocab_size + 1
num_outputs = envs[0].num_actions
question_padding_token = envs[0].question_padding_token
action_padding_token = envs[0].action_padding_token
q1 = TransformerEncoderModel(num_question_tokens=num_question_tokens, num_outputs=num_outputs,
                             question_padding_token=question_padding_token, action_padding_token=action_padding_token,
                             device=device)
q2 = TransformerEncoderModel(num_question_tokens=num_question_tokens, num_outputs=num_outputs,
                             question_padding_token=question_padding_token, action_padding_token=action_padding_token,
                             device=device)

# init replay buffer from trajectory cache on disk
replay_buffer = extract_replay_buffer_from_trajectory_cache(hparams.train.random_exploration_trajectory_cache_filepath,
                                                            hparams.train.replay_buffer_size,
                                                            hparams.train.gamma,
                                                            selected_filenames=hparams.env.selected_filenames)
replay_priority = np.ones(len(replay_buffer)) * hparams.train.default_replay_buffer_priority

# training loop --------------------------------------------------------------------------------------------------------

added_graphs = []
added_to_replay_buffer = 0
batch_i = last_fill_buffer_batch_i = last_eval_batch_i = last_target_network_update_batch_i = 0
for epoch_i in range(hparams.train.num_epochs):
    # sample and train -----------

    # sample from prioritized replay buffer
    replay_probability = replay_priority / np.sum(replay_priority)
    steps_to_sample = hparams.train.batch_size * hparams.train.batches_per_epoch
    sampled_train_idxs = np.random.choice(np.arange(len(replay_buffer)), size=steps_to_sample, p=replay_probability)
    sampled_steps = replay_buffer[sampled_train_idxs]

    # construct data loader
    step_dataset = StepDataset(sampled_steps, q1.device)
    data_loader = DataLoader(step_dataset, batch_size=q1.batch_size, shuffle=False, drop_last=True)

    # train
    print(f'({hparams.run.name}) batch #{batch_i}')
    print(len(replay_buffer))
    if random.random() < 0.5:
        batch_i, td_error_batches, batches = train(q1, q2, data_loader, writer, batch_i)
    else:
        batch_i, td_error_batches, batches = train(q2, q1, data_loader, writer, batch_i)

    # fill buffer -----------

    print(f'fresh replay buffer: {round(added_to_replay_buffer / len(replay_buffer) * 100, 2)}%')
    if batch_i >= hparams.train.num_batches_until_fill_buffer and \
            batch_i - last_fill_buffer_batch_i > hparams.train.batches_per_fill_buffer:
        last_fill_buffer_batch_i = batch_i
        latest_buffer, added_graphs = fill_buffer(q1, envs, trajectory_statistics, None)
        latest_buffer = add_trajectory_return_to_trajectories(latest_buffer, gamma=hparams.train.gamma)
        latest_buffer = np.array(flatten(latest_buffer))
        latest_replay_priority = np.ones(len(latest_buffer)) * hparams.train.default_replay_buffer_priority
        # add fresh experience to replay buffer
        replay_buffer = np.concatenate([replay_buffer, latest_buffer])
        replay_priority = np.concatenate([replay_priority, latest_replay_priority])
        added_to_replay_buffer += len(latest_buffer)
        fill_buffer_idxs = np.arange(len(replay_buffer)-len(latest_buffer), len(replay_buffer))
    else:
        fill_buffer_idxs = np.array([], dtype=np.int64)  # otherwise there are no indices corresponding to fresh experience

    # update replay priority -----------

    # sample indices for computing td error
    num_to_sample = hparams.train.n_batch_td_error * hparams.train.sample_td_error_batch_size - \
        len(sampled_train_idxs) - len(fill_buffer_idxs)
    sampled_idxs = np.random.choice(np.arange(len(replay_buffer)), size=num_to_sample)
    td_error_update_idxs = np.concatenate([sampled_train_idxs, fill_buffer_idxs, sampled_idxs])
    sampled_steps = replay_buffer[td_error_update_idxs]
    if random.random() < 0.5:
        td_error = get_td_error(q1, q2, sampled_steps)
    else:
        td_error = get_td_error(q2, q1, sampled_steps)

    # update replay priority
    replay_priority[td_error_update_idxs] = td_error.cpu().detach().numpy() ** hparams.train.prioritization_exponent
    # visualize_replay_priority(envs, replay_priority, replay_buffer)

    # eval -----------
    if batch_i - last_eval_batch_i >= hparams.train.batches_per_eval:
        last_eval_batch_i = batch_i
        mean_val_reward, eval_graphs = run_eval(q1, envs, writer, batch_i, hparams.train.n_required_validation_episodes)
        # log batches
        log_to_text_file(f'\nbatch #{batch_i}', logging_batches_filepath)
        log_batches(batches, td_error_batches, envs[0], logging_batches_filepath)
        # log added graphs
        if len(added_graphs) > 0:
            log_to_text_file("added graphs:", logging_batches_filepath)
            selected_added_graphs = random.choices(added_graphs, k=min(len(added_graphs), 10))
            selected_added_graphs_string = "\n".join(selected_added_graphs)
            log_to_text_file(selected_added_graphs_string, logging_batches_filepath)
        # log eval trajectories
        log_to_text_file("validation graphs:", logging_batches_filepath)
        selected_eval_graphs = random.choices(eval_graphs, k=min(len(eval_graphs), 10))
        selected_eval_graphs_string = "\n".join(selected_eval_graphs)
        log_to_text_file(selected_eval_graphs_string, logging_batches_filepath)
        # log eval reward
        log_to_text_file(f'mean val reward: {mean_val_reward}', logging_batches_filepath)

