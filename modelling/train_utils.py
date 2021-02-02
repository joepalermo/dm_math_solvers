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
    obs_batch = np.concatenate(obs_batch)
    return obs_batch, envs_info


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


def get_action_batch(obs_batch, envs, model=None, eval=False):
    # get model output
    if model:
        obs_batch = torch.from_numpy(obs_batch.astype(np.int64))
        output_batch = model(obs_batch.to(model.device)).detach().cpu().numpy()
        model_type = hparams.model.model_type
    else:
        output_batch = np.random.uniform(size=(len(obs_batch),len(envs[0].actions)))
        model_type = 'policy'
    if model_type == 'policy':
        output_batch = softmax(output_batch, axis=1)
    actions = []
    for i, env in enumerate(envs):
        # mask the corresponding model output
        masked_output = env.mask_invalid_types(output_batch[i])
        if model_type == 'policy':
            # normalize and sample
            masked_policy_vector = masked_output
            masked_normed_policy_vector = masked_policy_vector / np.sum(masked_policy_vector)
            action_index = np.random.choice(env.action_indices, p=masked_normed_policy_vector)
        elif model_type == 'value':
            eps_ = random.random()
            if eps_ < model.epsilon and not eval:
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
def get_policy(model, obs):
    from torch.distributions.categorical import Categorical
    logits = model(obs)
    return Categorical(logits=logits)


def vpg_loss(model, obs, act, weights):
    # make loss function whose gradient, for the right data, is policy gradient
    logp = get_policy(model, obs).log_prob(act)
    return -(logp * weights).mean()


def vpg_step(model, state_batch, action_batch, reward_batch):
    # take a single policy gradient update step
    model.optimizer.zero_grad()
    batch_loss = vpg_loss(model=model, obs=state_batch, act=action_batch, weights=reward_batch)
    batch_loss.backward()
    # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), model.max_grad_norm)
    model.optimizer.step()
    return batch_loss


def dqn_step(model, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
    # Take a single deep Q learning update step
    targets = reward_batch + (1 - done_batch) * hparams.train.gamma * torch.max(model(next_state_batch), dim=1)[0]
    model.optimizer.zero_grad()
    batch_output = model(state_batch)
    batch_output = batch_output.gather(1, action_batch.view(-1,1)).squeeze()
    td_error = torch.abs(targets - batch_output)
    batch_loss = torch.nn.MSELoss()(batch_output, targets)
    batch_loss.backward()
    # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), model.max_grad_norm)
    model.optimizer.step()
    return batch_loss, td_error


class StepDataset(torch.utils.data.Dataset):
    """Step Dataset"""

    def __init__(self, steps, device):
        self.steps = steps
        self.device = device

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, idx):
        # return only the (state, action, reward)?
        state, action, reward, next_state, done = self.steps[idx]
        state = torch.from_numpy(state.astype(np.int64)).to(self.device)
        action = torch.from_numpy(np.array(action, dtype=np.int64)).to(self.device)
        reward = torch.from_numpy(np.array(reward, dtype=np.int64)).to(self.device)
        next_state = torch.from_numpy(next_state.astype(np.int64)).to(self.device)
        done = torch.from_numpy(np.array(done, dtype=np.int64)).to(self.device)
        return state, action, reward, next_state, done


def fill_buffer(model, envs, trajectory_statistics):
    '''
    :param model:
    :param envs:
    :param trajectory_statistics:
    :return:
    '''
    assert hparams.train.mode == 'positive_only' or hparams.train.mode == 'balanced'
    # reset all environments
    cached_steps = 0
    trajectory_buffer = []
    buffer_positives = 1
    buffer_negatives = 1  # init to 1 to prevent division by zero
    # init trajectory cache from storage
    trajectory_cache = SqliteDict(hparams.env.trajectory_cache_filepath, autocommit=True)
    obs_batch, envs_info = reset_all(envs, trajectory_statistics=trajectory_statistics, train=True)
    # take steps in all environments until the number of cached steps reaches a threshold
    while cached_steps < hparams.train.buffer_threshold:
        # take a step in each environment in "parallel"
        action_batch = get_action_batch(obs_batch, envs, model=model)
        obs_batch, step_batch = step_all(envs, action_batch)
        # for each environment process the most recent step
        for env_i, ((obs, reward, done, info), action) in enumerate(zip(step_batch, action_batch)):
            # cache the latest step from each environment
            envs_info[env_i]['trajectory'].append((obs.astype(np.int16), action, reward, done, info))
            if done:
                # print(envs_info[env_i]['trajectory'][-1][4]['raw_observation'])
                def positive_condition(buffer_positives, buffer_negatives, reward):
                    return (hparams.train.mode == 'positive_only' and reward == 1) or \
                        (hparams.train.mode == 'balanced' and \
                        buffer_positives / buffer_negatives <= hparams.train.positive_to_negative_ratio and \
                        reward == 1)
                def negative_condition(buffer_positives, buffer_negatives, reward):
                    return hparams.train.mode == 'balanced' and \
                        buffer_positives / buffer_negatives > hparams.train.positive_to_negative_ratio and \
                        reward == -1
                # check if conditions to cache the trajectory are met
                if positive_condition(buffer_positives, buffer_negatives, reward) or \
                        negative_condition(buffer_positives, buffer_negatives, reward):
                    aligned_trajectory = align_trajectory(envs_info[env_i]['trajectory'])
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
                # # append first state of trajectory after reset
                # info_dict = {'raw_observation': envs_info[env_i]['question']}
                # envs_info[env_i]['trajectory'].append((obs_batch[env_i].astype(np.int16), None, None, None, info_dict))
    trajectory_cache.close()
    return trajectory_buffer


def train(model, data_loader, writer, current_batch_i):
    model.train()
    td_errors = list()
    for i, (state_batch, action_batch, reward_batch, next_state_batch, done_batch) in enumerate(data_loader):
        batch_loss, td_error = dqn_step(model, state_batch, action_batch, reward_batch, next_state_batch, done_batch)
        # batch_loss = vpg_step(model, state_batch, action_batch, reward_batch)
        writer.add_scalar('Train/loss', batch_loss, current_batch_i)
        # writer.add_scalar('Train/gradients', grad_norm, current_batch_i)
        current_batch_i += 1
        td_errors.append(td_error)
    td_error = torch.cat(td_errors)
    return current_batch_i, td_error


def run_eval(model, envs, writer, batch_i, n_required_validation_episodes):
    model.eval()
    logdir = get_logdir()
    total_reward = {}  # key: (module_name, difficulty) val: dict[key: n_completed_episodes or tot_reward]
    n_completed_validation_episodes = 0
    obs_batch, envs_info = reset_all(envs, train=False)
    while True:
        # take a step in each environment in "parallel"
        with torch.no_grad():
            action_batch = get_action_batch(obs_batch, envs, model=model, eval=True)
        obs_batch, step_batch = step_all(envs, action_batch)
        # for each environment process the most recent step
        for env_i, ((obs, reward, done, info), action) in enumerate(zip(step_batch, action_batch)):
            envs_info[env_i]['trajectory'].append((obs.astype(np.int16), action, reward, done, info))
            # if episode is complete, check if trajectory should be kept in buffer and reset environment
            if done:
                k = (envs[env_i].module_name, envs[env_i].difficulty)
                # TODO: remove random print
                if random.random() < 0.1:
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

                obs_batch[env_i], envs_info[env_i] = reset_environment(envs[env_i], train=False)
        if n_completed_validation_episodes > n_required_validation_episodes:
            break
    all_modules_reward = 0
    for k in total_reward.keys():
        mean_val_reward_per_module = total_reward[k]["tot_reward"] / total_reward[k]["n_completed_validation_episodes"]
        all_modules_reward += total_reward[k]["tot_reward"]
        writer.add_scalar(f'Val/{k[0]}_{k[1]}_reward', mean_val_reward_per_module, batch_i)

    mean_val_reward = all_modules_reward / n_completed_validation_episodes
    # check whether LR should be annealed via ReduceLROnPlateau
    model.scheduler.step(mean_val_reward)
    writer.add_scalar('Train/lr', model.optimizer.param_groups[0]['lr'], batch_i)
    writer.add_scalar('Val/tot_reward', mean_val_reward, batch_i)
    print(f'{batch_i} batches completed, mean validation reward: {mean_val_reward}')
    writer.close()