import copy
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.special import softmax
from environment.envs import MathEnv
from modelling.cache_utils import align_trajectory, cache_trajectory
from utils import flatten
from hparams import HParams
hparams = HParams.get_hparams_by_name('rl_math')
from sqlitedict import SqliteDict


def get_logdir():
    return f'logs-{hparams.run.name}'


def init_trajectory_data_structures(env):
    '''define data structures to track correct graphs'''
    trajectory_statistics = {}
    for module_name in env.train.keys():
        for difficulty in env.train[module_name].keys():
            if (module_name, difficulty) not in trajectory_statistics:
                trajectory_statistics[(module_name, difficulty)] = 0
    return trajectory_statistics


def init_envs(env_config):
    env = MathEnv(env_config)
    envs = [env]
    envs.extend([copy.copy(env) for _ in range(1, env_config.num_environments)])
    return envs


def reset_all(envs, trajectory_statistics=None, train=True):
    '''if trajectory_statistics is not None then select the module_name and difficulty which has been
    least rewarded thus far, else select module_name and difficulty randomly.'''
    envs_info = []
    obs_batch = []
    prev_actions_batch = []
    for env in envs:
        if trajectory_statistics is not None:
            module_name, difficulty = min(trajectory_statistics, key=trajectory_statistics.get)
            obs, info = env.reset_by_module_and_difficulty(module_name, difficulty, train=train)
        else:
            obs, info = env.reset(train=train)
        envs_info.append({'question': info['raw_observation'],
                          'trajectory': [(obs, None, None, None, info)],
                          'module_name': env.module_name,
                          'difficulty': env.difficulty,
                          'module_difficulty_index': env.module_difficulty_index})
        obs_batch.append(np.expand_dims(obs, 0))
        # prev_action_batch is initialized to contain only the padding action
        prev_actions_batch.append(np.expand_dims([env.num_actions+1 for _ in range(env.max_num_nodes)], 0))
    obs_batch = np.concatenate(obs_batch)
    prev_actions_batch = np.concatenate(prev_actions_batch)
    return obs_batch, prev_actions_batch, envs_info


def step_all(envs, action_batch):
    step_batch = list()
    obs_batch = list()
    for env, action in zip(envs, action_batch):
        step = env.step(action)
        (obs, reward, done, info) = step
        step_batch.append(step)
        obs_batch.append(np.expand_dims(obs, 0))
    obs_batch = np.concatenate(obs_batch)
    return obs_batch, step_batch


def get_action_batch(obs_batch, prev_actions_batch, envs, network=None, eval=False):
    # get network output
    if network:
        obs_batch = torch.from_numpy(obs_batch.astype(np.int64)).to(network.device)
        prev_actions_batch = torch.from_numpy(prev_actions_batch.astype(np.int64)).to(network.device)
        output_batch = network(obs_batch, prev_actions_batch).detach().cpu().numpy()
        model_type = hparams.model.model_type
    else:
        output_batch = np.random.uniform(size=(len(obs_batch),len(envs[0].actions)))
        model_type = 'policy'
    actions = []
    for i, env in enumerate(envs):
        # mask the corresponding network output
        masked_output = env.mask_invalid_types(output_batch[i])
        if model_type == 'policy':
            # normalize and sample
            masked_policy_vector = masked_output
            masked_normed_policy_vector = masked_policy_vector / np.sum(masked_policy_vector)
            action_index = np.random.choice(env.action_indices, p=masked_normed_policy_vector)
        elif model_type == 'value':
            eps_ = random.random()
            if eps_ < network.epsilon and not eval:
                # take random action from among unmasked actions
                available_actions = [i for i in env.action_indices if masked_output[i] != 0]
                action_index = random.choice(available_actions)
            else:
                action_index = np.argmax(masked_output)
        actions.append(action_index)
    return actions


def reset_environment(env, train=True):
    obs, info = env.reset(train=train)
    return obs, {'question': info['raw_observation'],
                 'trajectory': [(obs, None, None, None, None)],
                 'module_name': env.module_name,
                 'difficulty': env.difficulty,
                 'module_difficulty_index': env.module_difficulty_index}


def reset_environment_with_least_rewarded_problem_type(env, trajectory_statistics, train=True):
    module_name, difficulty = min(trajectory_statistics, key=trajectory_statistics.get)
    obs, info = env.reset_by_module_and_difficulty(module_name, difficulty, train=train)
    return obs, {'question': info['raw_observation'],
                 'trajectory': [(obs, None, None, None, None)],
                 'module_name': module_name,
                 'difficulty': difficulty,
                 'module_difficulty_index': env.module_difficulty_index}


# make function to compute action distribution
def get_policy(network, obs):
    from torch.distributions.categorical import Categorical
    logits = network(obs)
    return Categorical(logits=logits)


def vpg_loss(network, obs, act, weights):
    # make loss function whose gradient, for the right data, is policy gradient
    logp = get_policy(network, obs).log_prob(act)
    return -(logp * weights).mean()


def vpg_step(network, state_batch, action_batch, reward_batch):
    # take a single policy gradient update step
    network.optimizer.zero_grad()
    batch_loss = vpg_loss(network=network, obs=state_batch, act=action_batch, weights=reward_batch)
    batch_loss.backward()
    # grad_norm = torch.nn.utils.clip_grad_norm_(network.parameters(), network.max_grad_norm)
    network.optimizer.step()
    return batch_loss


def dqn_step(network, target_network, batch):
    state_batch, action_batch, reward_batch, next_state_batch, prev_actions_batch, done_batch = batch
    # compute the target
    if target_network is None:
        targets = reward_batch + (1 - done_batch) * hparams.train.gamma * \
                  torch.max(network(next_state_batch, prev_actions_batch), dim=1)[0]
    else:
        with torch.no_grad():
            targets = reward_batch + (1 - done_batch) * hparams.train.gamma * \
                      torch.max(target_network(next_state_batch, prev_actions_batch), dim=1)[0]
    network.optimizer.zero_grad()
    batch_output = network(state_batch, prev_actions_batch)
    batch_output = batch_output.gather(1, action_batch.view(-1,1)).squeeze()
    td_error = torch.abs(targets - batch_output)
    batch_loss = torch.nn.MSELoss()(batch_output, targets)
    batch_loss.backward()
    # grad_norm = torch.nn.utils.clip_grad_norm_(network.parameters(), network.max_grad_norm)
    network.optimizer.step()
    return batch_loss, td_error

def get_td_error(network, sampled_steps):
    step_dataset = StepDataset(sampled_steps, network.device)
    data_loader = DataLoader(step_dataset, batch_size=hparams.train.sample_td_error_batch_size, shuffle=False, drop_last=True)
    td_error_list = list()
    for batch in data_loader:
        with torch.no_grad():
            state_batch, action_batch, reward_batch, next_state_batch, prev_actions_batch, done_batch = batch
            targets = reward_batch + (1 - done_batch) * hparams.train.gamma * \
                      torch.max(network(next_state_batch, prev_actions_batch), dim=1)[0]
            batch_output = network(state_batch, prev_actions_batch)
            batch_output = batch_output.gather(1, action_batch.view(-1, 1)).squeeze()
            batch_td_error = torch.abs(targets - batch_output)
            td_error_list.append(batch_td_error)
    return torch.cat(td_error_list)


class StepDataset(torch.utils.data.Dataset):
    """Step Dataset"""

    def __init__(self, steps, device):
        self.steps = steps
        self.device = device

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, idx):
        state, action, reward, next_state, prev_actions, done = self.steps[idx]
        state = torch.from_numpy(state.astype(np.int64)).to(self.device)
        action = torch.from_numpy(np.array(action, dtype=np.int64)).to(self.device)
        reward = torch.from_numpy(np.array(reward, dtype=np.int64)).to(self.device)
        next_state = torch.from_numpy(next_state.astype(np.int64)).to(self.device)
        prev_actions = torch.from_numpy(np.array(prev_actions, dtype=np.int64)).to(self.device)
        done = torch.from_numpy(np.array(done, dtype=np.int64)).to(self.device)
        return state, action, reward, next_state, prev_actions, done


def update_prev_actions(prev_actions_batch, action_batch, action_padding_token):
    last_padding_index = np.array([np.where(prev_actions_batch[i] == action_padding_token)[0].min()
                            for i in range(len(prev_actions_batch))])
    prev_actions_batch[np.arange(len(prev_actions_batch)), last_padding_index] = action_batch
    return prev_actions_batch


def fill_buffer(network, envs, trajectory_statistics, trajectory_cache_filepath):
    '''
    :param network:
    :param envs:
    :param trajectory_statistics:
    :return:
    '''
    assert hparams.train.fill_buffer_mode == 'positive_only' or \
           hparams.train.fill_buffer_mode == 'balanced' or \
           hparams.train.fill_buffer_mode == 'anything'
    # reset all environments
    cached_steps = 0
    trajectory_buffer = []
    buffer_positives = 1
    buffer_negatives = 1  # init to 1 to prevent division by zero
    if trajectory_cache_filepath is not None:
        # init trajectory cache from storage
        trajectory_cache = SqliteDict(trajectory_cache_filepath, autocommit=True)
    obs_batch, prev_actions_batch, envs_info = reset_all(envs, trajectory_statistics=trajectory_statistics, train=True)
    # take steps in all environments until the number of cached steps reaches a threshold
    while cached_steps < hparams.train.buffer_threshold:
        # take a step in each environment in "parallel"
        with torch.no_grad():
            action_batch = get_action_batch(obs_batch, prev_actions_batch, envs, network=network)
        obs_batch, step_batch = step_all(envs, action_batch)
        prev_actions_batch = update_prev_actions(prev_actions_batch, action_batch,
                                                 action_padding_token=envs[0].num_actions+1)
        # for each environment process the most recent step
        for env_i, ((obs, reward, done, info), action) in enumerate(zip(step_batch, action_batch)):
            # cache the latest step from each environment
            envs_info[env_i]['trajectory'].append((obs.astype(np.int16), action, reward, done, info))
            if done:
                # if random.random() < 0.01:
                #     print(f"{info['raw_observation']} = {envs[env_i].compute_graph.eval()}, reward: {reward}\n")
                # print(envs_info[env_i]['trajectory'][-1][4]['raw_observation'])
                def positive_condition(buffer_positives, buffer_negatives, reward):
                    return (hparams.train.fill_buffer_mode == 'positive_only' and reward == 1) or \
                           (hparams.train.fill_buffer_mode == 'balanced' and \
                                buffer_positives / buffer_negatives <= hparams.train.positive_to_negative_ratio and \
                                reward == 1)
                def negative_condition(buffer_positives, buffer_negatives, reward):
                    return hparams.train.fill_buffer_mode == 'balanced' and \
                        buffer_positives / buffer_negatives > hparams.train.positive_to_negative_ratio and \
                        reward == 0
                # check if conditions to cache the trajectory are met
                if positive_condition(buffer_positives, buffer_negatives, reward) or \
                        negative_condition(buffer_positives, buffer_negatives, reward) or \
                        hparams.train.fill_buffer_mode == 'anything':
                    aligned_trajectory = align_trajectory(envs_info[env_i]['trajectory'],
                                                          action_start_token=envs[env_i].num_actions,
                                                          action_padding_token=envs[env_i].num_actions+1,
                                                          max_num_nodes=envs[env_i].max_num_nodes)
                    if trajectory_cache_filepath is not None:
                        cache_trajectory(envs_info[env_i], aligned_trajectory, trajectory_cache)
                    trajectory_buffer.append(aligned_trajectory)
                    cached_steps += len(aligned_trajectory)
                    with open(f'{get_logdir()}/training_graphs.txt', 'a') as f:
                        f.write(f"{info['raw_observation']} = {envs[env_i].compute_graph.eval()}\n")
                    trajectory_statistics[(envs_info[env_i]['module_name'], envs_info[env_i]['difficulty'])] += 1
                if positive_condition(buffer_positives, buffer_negatives, reward):
                    buffer_positives += 1
                elif negative_condition(buffer_positives, buffer_negatives, reward):
                    buffer_negatives += 1
                # reset environment
                obs_batch[env_i], envs_info[env_i] = \
                    reset_environment_with_least_rewarded_problem_type(envs[env_i], trajectory_statistics,
                                                                       train=True)
                prev_actions_batch[env_i] = np.array([envs[env_i].num_actions+1
                                                      for _ in range(envs[env_i].max_num_nodes)])
                # # append first state of trajectory after reset
                # info_dict = {'raw_observation': envs_info[env_i]['question']}
                # envs_info[env_i]['trajectory'].append((obs_batch[env_i].astype(np.int16), None, None, None, info_dict))
    if trajectory_cache_filepath is not None:
        trajectory_cache.close()
    return trajectory_buffer


def train(network, target_network, data_loader, writer, current_batch_i):
    network.train()
    td_errors = list()
    losses = list()
    batches = list()
    for i, batch in enumerate(data_loader):
        batch_loss, td_error = dqn_step(network, target_network, batch)
        writer.add_scalar('Train/loss', batch_loss, current_batch_i)
        # writer.add_scalar('Train/gradients', grad_norm, current_batch_i)
        current_batch_i += 1
        td_errors.append(td_error)
        losses.append(float(batch_loss.detach().cpu().numpy()))
        batches.append(batch[:2])
    td_error = torch.cat(td_errors)
    mean_batch_loss = np.array(losses).mean()
    print(f'mean_batch_loss: {mean_batch_loss}')
    return current_batch_i, td_error, batches


def run_eval(network, envs, writer, batch_i, n_required_validation_episodes):
    network.eval()
    logdir = get_logdir()
    total_reward = {}  # key: (module_name, difficulty) val: dict[key: n_completed_episodes or tot_reward]
    n_completed_validation_episodes = 0
    obs_batch, prev_actions_batch, envs_info = reset_all(envs, train=False)
    while True:
        # take a step in each environment in "parallel"
        with torch.no_grad():
            action_batch = get_action_batch(obs_batch, prev_actions_batch, envs, network=network, eval=True)
        obs_batch, step_batch = step_all(envs, action_batch)
        prev_actions_batch = update_prev_actions(prev_actions_batch, action_batch,
                                                 action_padding_token=envs[0].num_actions+1)
        # for each environment process the most recent step
        for env_i, ((obs, reward, done, info), action) in enumerate(zip(step_batch, action_batch)):
            envs_info[env_i]['trajectory'].append((obs.astype(np.int16), action, reward, done, info))
            # if episode is complete, check if trajectory should be kept in buffer and reset environment
            if done:
                k = (envs[env_i].module_name, envs[env_i].difficulty)
                if random.random() < 0.05:
                    print(f"{info['raw_observation']} = {envs[env_i].compute_graph.eval()}, reward: {reward}\n")
                with open(f'{logdir}/validation_graphs_{k[0]}_{k[1]}.txt', 'a') as f:
                    f.write(f"{info['raw_observation']} = {envs[env_i].compute_graph.eval()}, reward: {reward}\n")
                if k in total_reward:
                    total_reward[k]["n_completed_validation_episodes"] += 1
                    total_reward[k]["tot_reward"] += reward
                else:
                    total_reward[k] = {}
                    total_reward[k]["n_completed_validation_episodes"] = 1
                    total_reward[k]["tot_reward"] = reward
                n_completed_validation_episodes += 1
                # reset environment
                obs_batch[env_i], envs_info[env_i] = reset_environment(envs[env_i], train=False)
                prev_actions_batch[env_i] = np.array([envs[env_i].num_actions+1
                                                      for _ in range(envs[env_i].max_num_nodes)])
        if n_completed_validation_episodes > n_required_validation_episodes:
            break
    all_modules_reward = 0
    for k in total_reward.keys():
        mean_val_reward_per_module = total_reward[k]["tot_reward"] / total_reward[k]["n_completed_validation_episodes"]
        all_modules_reward += total_reward[k]["tot_reward"]
        writer.add_scalar(f'Val/{k[0]}_{k[1]}_reward', mean_val_reward_per_module, batch_i)

    mean_val_reward = all_modules_reward / n_completed_validation_episodes
    # check whether LR should be annealed via ReduceLROnPlateau
    network.scheduler.step(mean_val_reward)
    writer.add_scalar('Train/lr', network.optimizer.param_groups[0]['lr'], batch_i)
    writer.add_scalar('Val/tot_reward', mean_val_reward, batch_i)
    print(f'{batch_i} batches completed, mean validation reward: {mean_val_reward}')
    writer.close()
    return mean_val_reward


def visualize_replay_priority(envs, replay_priority, replay_buffer):
    num_samples = 1
    norm = np.sum(replay_priority)

    print('\n\thighest:')
    highest_priority_idxs = replay_priority.argsort()[-num_samples:]
    for idx in highest_priority_idxs:
        print(f"\n\tpriority: {replay_priority[idx]}, probability: {replay_priority[idx] / norm}")
        print(f"\tstate: {envs[0].decode(replay_buffer[idx][0])}")
        print(f"\taction: {replay_buffer[idx][1]}")
        print(f"\treward: {replay_buffer[idx][2]}")
        print(f"\tnext state: {envs[0].decode(replay_buffer[idx][3])}")

    print('\n\tlowest:')
    lowest_priority_idxs = replay_priority.argsort()[:num_samples]
    for idx in lowest_priority_idxs:
        print(f"\n\tpriority: {replay_priority[idx]}, probability: {replay_priority[idx] / norm}")
        print(f"\tstate: {envs[0].decode(replay_buffer[idx][0])}")
        print(f"\taction: {replay_buffer[idx][1]}")
        print(f"\treward: {replay_buffer[idx][2]}")
        print(f"\tnext state: {envs[0].decode(replay_buffer[idx][3])}")

    # print('\nrandom:')
    # random_idxs = np.random.choice(np.arange(len(replay_priority)), size=num_samples)
    # for idx in random_idxs:
    #     print(f"\npriority: {replay_priority[idx]}, probability: {replay_priority[idx]/norm}")
    #     print(f"state: {envs[0].decode(replay_buffer[idx][0])}")
    #     print(f"action: {replay_buffer[idx][1]}")
    #     print(f"reward: {replay_buffer[idx][2]}")

    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # sns.histplot(replay_priority / np.sum(replay_priority))
    # sns.histplot(replay_priority)
    # plt.show()
    # import time; time.sleep(1000)

