import copy
import math
import random

import numpy as np
import torch
from scipy.special import softmax
from environment.envs import MathEnv
# from train import batch_size, model, device, optimizer, max_grad_norm, writer


def init_trajectory_data_structures(env):
    '''define data structures to track correct graphs'''
    rewarded_trajectories = {}
    rewarded_trajectory_statistics = {}
    for module_name in env.train.keys():
        for difficulty in env.train[module_name].keys():
            if (module_name, difficulty) not in rewarded_trajectories:
                rewarded_trajectories[(module_name, difficulty)] = []
            if (module_name, difficulty) not in rewarded_trajectory_statistics:
                rewarded_trajectory_statistics[(module_name, difficulty)] = 0
    return rewarded_trajectories, rewarded_trajectory_statistics


def init_envs(env_config, num_environments=10):
    env = MathEnv(env_config)
    envs = [env]
    envs.extend([copy.copy(env) for _ in range(1, num_environments)])
    return envs


def reset_all(envs, rewarded_trajectory_statistics=None, train=True):
    '''if rewarded_trajectory_statistics is not None then select the module_name and difficulty which has been
    least rewarded thus far, else select module_name and difficulty randomly.'''
    envs_info = []
    obs_batch = []
    for env in envs:
        if rewarded_trajectory_statistics is not None:
            module_name, difficulty = min(rewarded_trajectory_statistics, key=rewarded_trajectory_statistics.get)
            obs, info = env.reset_by_module_and_difficulty(module_name, difficulty, train=train)
        else:
            obs, info = env.reset(train=train)
            module_name, difficulty = env.module_name, env.difficulty
        envs_info.append({'problem_statement': info['raw_observation'],
                          'trajectory': list(),
                          'module_name': module_name,
                          'difficulty': difficulty})
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


def get_action_batch(obs_batch, envs, model=None):
    if model:
        obs_batch = torch.from_numpy(obs_batch.astype(np.int64))
        logits_batch = model(obs_batch.to(model.device)).detach().cpu().numpy()
    else:
        logits_batch = np.random.uniform(size=(32,35))
    policy_batch = softmax(logits_batch, axis=1)
    actions = []
    for i, env in enumerate(envs):
        masked_policy_vector = env.mask_invalid_types(policy_batch[i])
        masked_normed_policy_vector = masked_policy_vector / np.sum(
            masked_policy_vector
        )
        action_index = np.random.choice(env.action_indices, p=masked_normed_policy_vector)
        actions.append(action_index)
    return actions


def update_trajectory_data_structures(env_info, rewarded_trajectories, rewarded_trajectory_statistics):
    module_name = env_info['module_name']
    difficulty = env_info['difficulty']
    trajectory = env_info['trajectory']
    reward = trajectory[-1][2]
    if reward == 1:
        rewarded_trajectories[(module_name, difficulty)].append(trajectory)
        rewarded_trajectory_statistics[(module_name, difficulty)] += 1


def reset_environment(env, train=True):
    obs, info = env.reset(train=train)
    return obs, {'problem_statement': info['raw_observation'],
                 'trajectory': [(obs, None, None, None, None)],
                 'module_name': env.module_name,
                 'difficulty': env.difficulty}


def reset_environment_with_least_rewarded_problem_type(env, rewarded_trajectory_statistics, train=True):
    module_name, difficulty = min(rewarded_trajectory_statistics, key=rewarded_trajectory_statistics.get)
    obs, info = env.reset_by_module_and_difficulty(module_name, difficulty, train=train)
    return obs, {'problem_statement': info['raw_observation'],
                 'trajectory': [(obs, None, None, None, None)],
                 'module_name': module_name,
                 'difficulty': difficulty}


def extract_buffer_trajectory(raw_trajectory, reward):
    states = [state for state, _, _, _, _ in raw_trajectory[0:-1]]
    action_reward = [(action, reward) for _, action, _, _, _ in raw_trajectory[1:]]
    buffer_trajectory = [(state, action, reward) for state, (action, reward) in zip(states, action_reward)]
    return buffer_trajectory


def inspect_performance(trajectories, rewarded_trajectory_statistics):
    for module_name, difficulty in trajectories.keys():
        if len(trajectories[(module_name,difficulty)]) > 0:
            percentage_correct = rewarded_trajectory_statistics[(module_name,difficulty)] / len(trajectories[(module_name,difficulty)]) * 100
            print(f"{module_name}@{difficulty}: {rewarded_trajectory_statistics[(module_name,difficulty)]} / {len(trajectories[(module_name,difficulty)])} = {round(percentage_correct, 5)}%")


def train_on_buffer(model, buffer, writer, current_batch_i):
    n_batches = math.floor(len(buffer) / model.batch_size)
    random.shuffle(buffer)
    model.train()
    for buffer_batch_i in range(n_batches):
        batch = buffer[buffer_batch_i * model.batch_size: (buffer_batch_i + 1) * model.batch_size]
        state_batch = torch.from_numpy(
            np.concatenate([np.expand_dims(step[0], 0) for step in batch]).astype(np.int64)).to(model.device)
        action_batch = torch.from_numpy(
            np.concatenate([np.expand_dims(step[1], 0) for step in batch]).astype(np.int64)).to(model.device)
        reward_batch = torch.from_numpy(
            np.concatenate([np.expand_dims(step[2], 0) for step in batch]).astype(np.int64)).to(model.device)
        batch_logits = model(state_batch)
        batch_probs = torch.softmax(batch_logits, axis=1)
        # loss is given by -mean(log(model(a=a_t|s_t)) * R_t)
        loss = -torch.mean(torch.log(batch_probs[:, action_batch]) * reward_batch)
        # backprop + gradient descent
        model.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), model.max_grad_norm)
        model.optimizer.step()
        # increment the batch index + track the loss and gradients
        current_batch_i += 1
        writer.add_scalar('Train/loss', loss, current_batch_i)
        writer.add_scalar('Train/gradients', grad_norm, current_batch_i)
    return n_batches


def run_eval(model, envs, writer, batch_i, n_required_validation_episodes):
    model.eval()
    total_reward = {}  # key: (module_name, difficulty) val: dict[key: n_completed_episodes or tot_reward]
    n_completed_validation_episodes = 0
    obs_batch, envs_info = reset_all(envs, train=False)
    while True:
        # take a step in each environment in "parallel"
        with torch.no_grad():
            action_batch = get_action_batch(obs_batch, envs, model=model)
        obs_batch, step_batch = step_all(envs, action_batch)
        # for each environment process the most recent step
        for env_i, ((obs, reward, done, info), action) in enumerate(zip(step_batch, action_batch)):
            envs_info[env_i]['trajectory'].append((obs.astype(np.int16), action, reward, done, info))
            # if episode is complete, check if trajectory should be kept in buffer and reset environment
            if done:
                k = (envs[env_i].module_name, envs[env_i].difficulty)
                with open(f'artifacts/validation_graphs_{k[0]}_{k[1]}.txt', 'a') as f:
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


def fill_buffer(model, envs, buffer_threshold, positive_to_negative_ratio, rewarded_trajectories,
                rewarded_trajectory_statistics, verbose=False):
    # reset all environments
    buffer = []
    buffer_positives = 1
    buffer_negatives = 1  # init to 1 to prevent division by zero
    obs_batch, envs_info = reset_all(envs, rewarded_trajectory_statistics=rewarded_trajectory_statistics, train=True)
    # take steps in all environments num_parallel_steps times
    while True:
        # take a step in each environment in "parallel"
        action_batch = get_action_batch(obs_batch, envs, model=model)
        obs_batch, step_batch = step_all(envs, action_batch)
        # for each environment process the most recent step
        for env_i, ((obs, reward, done, info), action) in enumerate(zip(step_batch, action_batch)):
            envs_info[env_i]['trajectory'].append((obs.astype(np.int16), action, reward, done, info))
            # if episode is complete, check if trajectory should be kept in buffer and reset environment
            if done:
                update_trajectory_data_structures(envs_info[env_i], rewarded_trajectories, rewarded_trajectory_statistics)
                with open('artifacts/training_graphs.txt', 'a') as f:
                    f.write(f"{info['raw_observation']} = {envs[env_i].compute_graph.eval()}\n")
                if reward == 1 and verbose:
                    print(f"{info['raw_observation']} = {envs[env_i].compute_graph.eval()}")
                if buffer_positives / buffer_negatives <= positive_to_negative_ratio and reward == 1:
                    buffer_trajectory = extract_buffer_trajectory(envs_info[env_i]['trajectory'], reward)
                    buffer.extend(buffer_trajectory)
                    buffer_positives += 1
                elif buffer_positives / buffer_negatives > positive_to_negative_ratio and reward == -1:
                    buffer_trajectory = extract_buffer_trajectory(envs_info[env_i]['trajectory'], reward)
                    buffer.extend(buffer_trajectory)
                    buffer_negatives += 1
                obs_batch[env_i], envs_info[env_i] = \
                    reset_environment_with_least_rewarded_problem_type(envs[env_i], rewarded_trajectory_statistics,
                                                                       train=True)
        if len(buffer) > buffer_threshold:
            break
    return buffer


def load_buffer(trajectories_filepath):
    from utils import read_pickle
    buffer = list()
    trajectories = read_pickle(trajectories_filepath)
    for module_name, difficulty in trajectories:
        trajectories_ = trajectories[(module_name, difficulty)]
        for trajectory in trajectories_:
            reward = trajectory[-1][2]
            processed_trajectory = extract_buffer_trajectory(trajectory, reward)
            buffer.extend(processed_trajectory)
    return buffer