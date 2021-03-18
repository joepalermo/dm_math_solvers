import os
import pprint
import random
import math
import torch
from sqlitedict import SqliteDict
from utils import flatten
import numpy as np


def align_trajectory(raw_trajectory):
    states = [state for state, _, _, _, _ in raw_trajectory[:-1]]
    everything_else = [(next_state, action, reward, done) for next_state, action, reward, done, _ in raw_trajectory[1:]]
    aligned_trajectory = [(state, action, reward, next_state, done)
                         for state, (next_state, action, reward, done)
                         in zip(states, everything_else)]
    return aligned_trajectory


def cache_trajectory(env_info, aligned_trajectory, trajectory_cache):
    def same_problem_trajectory_equals(trajectory1, trajectory2):
        '''
        Since MathEnv is a deterministic environment, then if both trajectories are for the same problem and they have
        the same action sequence, they must be equal trajectories.

        Steps have the format: (state, action, reward, next_state, done)
        Therefore, actions can be accessed by indexing a step at position 1.
        '''
        return all([step1[1] == step2[1] for step1, step2 in zip(trajectory1, trajectory2)])
    raw_key = (env_info['module_name'], str(env_info['difficulty']),
               str(env_info['module_difficulty_index']))
    key = '-'.join(raw_key)
    if not key in trajectory_cache:
        trajectory_cache[key] = [aligned_trajectory]
    else:
        # if the key already exists in the trajectory_cache, then only cache the trajectory if it's not already present
        for aligned_trajectory_ in trajectory_cache[key]:
            if same_problem_trajectory_equals(aligned_trajectory_, aligned_trajectory):
                return
        # append the new trajectory
        trajectories = trajectory_cache[key]
        trajectories.append(aligned_trajectory)
        trajectory_cache[key] = trajectories


def extract_trajectory_cache(trajectory_cache_filepath, selected_filenames=None, verbose=False):
    all_trajectories = []
    selected_filenames = selected_filenames if selected_filenames is not None else []
    seleted_module_names = [fn.split('.txt')[0] for fn in selected_filenames]
    module_difficulty_trajectory_counts = {}
    try:
        trajectory_cache = SqliteDict(trajectory_cache_filepath, autocommit=True)
        for key in trajectory_cache:
            trajectories = trajectory_cache[key]
            module_difficulty = '-'.join(key.split('-')[:-1])
            module = module_difficulty.split('-')[0]
            if module_difficulty not in module_difficulty_trajectory_counts:
                module_difficulty_trajectory_counts[module_difficulty] = 1
            else:
                module_difficulty_trajectory_counts[module_difficulty] += 1
            if module in seleted_module_names:
                all_trajectories.extend(trajectories)
        if verbose:
            pprint.pprint(module_difficulty_trajectory_counts)
            print(f"# problems: {len(trajectory_cache)}")
            print(f"# trajectories: {len(all_trajectories)}")
            print(f"# steps: {len(flatten(all_trajectories))}")
    except:
        print(f"reading trajectory cache at {trajectory_cache_filepath} failed; trajectory cache may not exist.")
    return all_trajectories


def add_trajectory_return_to_trajectories(trajectories, gamma):
    mod_trajectories = []
    # add trajectory return to each step
    for trajectory in trajectories:
        trajectory_return = sum([reward*gamma**i for i, (_, _, reward, _, _) in enumerate(trajectory)])
        mod_trajectory = [(state, action, reward, next_state, done, trajectory_return)
            for state, action, reward, next_state, done in trajectory]
        mod_trajectories.append(mod_trajectory)
    return mod_trajectories


def extract_replay_buffer_from_trajectory_cache(trajectory_cache_filepath, replay_buffer_size, gamma,
                                                selected_filenames=None):
    trajectories = extract_trajectory_cache(trajectory_cache_filepath, selected_filenames=selected_filenames)
    trajectories = add_trajectory_return_to_trajectories(trajectories, gamma)
    replay_buffer = flatten(trajectories)
    random.shuffle(replay_buffer)
    return np.array(replay_buffer[:replay_buffer_size])


def visualize_trajectory_cache(decoder, trajectory_cache, num_to_sample=5):
    key_trajectory_pairs = random.sample(list(trajectory_cache.items()), min(num_to_sample, len(trajectory_cache)))
    print(f"size of trajectory cache: {len(trajectory_cache)}")
    for key, trajectories in key_trajectory_pairs:
        for trajectory in trajectories:
            last_state = trajectory[-1][3]
            reward = trajectory[-1][2]
            print("\t", decoder(last_state), f"reward: {reward}")


def visualize_trajectory_cache_by_module_and_difficulty(decoder, trajectory_cache, question_length,
                                                        num_to_sample=5):
    all_trajectories = {}
    for key in trajectory_cache:
        trajectories = trajectory_cache[key]
        module_difficulty = '-'.join(key.split('-')[:-1])
        if module_difficulty not in all_trajectories:
            all_trajectories[module_difficulty] = []
        else:
            all_trajectories[module_difficulty].extend(trajectories)
    module_difficulty_counts = {}
    total_num_steps = total_num_trajectories = 0
    for module_difficulty in all_trajectories:
        module_difficulty_trajectories = all_trajectories[module_difficulty]
        sampled_trajectories = [] if len(module_difficulty_trajectories) == 0 else \
            np.random.choice(module_difficulty_trajectories, size=num_to_sample)
        print(f"{module_difficulty} samples:")
        for trajectory in sampled_trajectories:
            question = decoder(trajectory[-1][3][:question_length])
            prev_actions = trajectory[-1][3][question_length:]
            reward = trajectory[-1][2]
            print(f'question: {question}, actions: {prev_actions}, reward: {reward}')
        print(f'{module_difficulty}')
        num_trajectories = len(all_trajectories[module_difficulty])
        num_steps = len(flatten(all_trajectories[module_difficulty]))
        print(f'# trajectories: {num_trajectories}')
        print(f'# steps: {num_steps}')
        module_difficulty_counts[module_difficulty] = (num_steps, num_trajectories)
        total_num_trajectories += num_trajectories
        total_num_steps += num_steps
    pprint.pprint(module_difficulty_counts)
    print(f'# total trajectories: {total_num_trajectories}')
    print(f'# total steps: {total_num_steps}')


def log_batches(batches, td_error_batches, env, filepath, num_batches=20):
    strings = []
    td_errors = []
    for batch, td_error_batch in zip(batches, td_error_batches):
        state_batch, action_batch, reward_batch, _, _, _ = batch
        for state, action, reward, td_error in zip(state_batch, action_batch, reward_batch, td_error_batch):
            decoded_state = env.decode(state[:env.max_sequence_length])
            actions = state[env.max_sequence_length:]
            # TODO: cleanup logging of actions
            step_string = 'td_error: {:.2f}, '.format(td_error) + f'{decoded_state}, {actions}' \
                f'action: {env.action_names[action]}, reward: {reward}'
            strings.append(step_string)
            td_errors.append(td_error)
    highest_td_errors_idxs = np.argsort(np.array(td_errors))[::-1][:num_batches].tolist()
    lowest_td_errors_idxs = np.argsort(np.array(td_errors))[:num_batches].tolist()
    highest_and_lowest = [strings[i] for i in highest_td_errors_idxs] + [strings[i] for i in lowest_td_errors_idxs]
    log_batches_string = "\n".join(highest_and_lowest)
    log_to_text_file(log_batches_string, filepath)


def log_to_text_file(string, filepath):
    if os.path.isfile(filepath):
        mode = 'a'
    else:
        mode = 'w'
    with open(filepath, mode) as f:
        f.write(string + '\n')
