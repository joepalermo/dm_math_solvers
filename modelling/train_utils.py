import copy
import numpy as np
import torch
from scipy.special import softmax
from environment.envs import MathEnv


def init_trajectory_data_structures(env):
    '''define data structures to track correct graphs'''
    rewarded_trajectory_statistics = {}
    for module_name in env.train.keys():
        for difficulty in env.train[module_name].keys():
            if (module_name, difficulty) not in rewarded_trajectory_statistics:
                rewarded_trajectory_statistics[(module_name, difficulty)] = 0
    return rewarded_trajectory_statistics


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


def get_action_batch(obs_batch, envs, device=None, model=None):
    if model:
        obs_batch = torch.from_numpy(obs_batch.astype(np.int64))
        logits_batch = model(obs_batch.to(device)).detach().cpu().numpy()
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


def update_rewarded_trajectory_statistics(env_info, rewarded_trajectory_statistics):
    module_name = env_info['module_name']
    difficulty = env_info['difficulty']
    trajectory = env_info['trajectory']
    reward = trajectory[-1][2]
    if reward == 1:
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