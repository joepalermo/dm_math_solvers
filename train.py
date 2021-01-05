import copy
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from utils import read_text_file, write_pickle
from scipy.special import softmax
from environment.envs import MathEnv
from modelling.transformer_encoder import TransformerEncoderModel
import math
import random

torch.cuda.set_device(0)

def init_trajectory_data_structures(env):
    '''define data structures to track correct graphs'''
    trajectories = {}
    rewarded_trajectory_statistics = {}
    for module_name in env.train.keys():
        for difficulty in env.train[module_name].keys():
            if (module_name, difficulty) not in trajectories and difficulty <= max_difficulty_level:
                trajectories[(module_name,difficulty)] = []
            if (module_name, difficulty) not in rewarded_trajectory_statistics and difficulty <= max_difficulty_level:
                rewarded_trajectory_statistics[(module_name, difficulty)] = 0
    return trajectories, rewarded_trajectory_statistics


def init_envs(env_config, num_environments=10):
    env = MathEnv(env_config)
    envs = [env]
    envs.extend([copy.copy(env) for _ in range(1, num_environments)])
    return envs


def reset_all(envs, train=True):
    envs_info = []
    obs_batch = []
    for env in envs:
        module_name, difficulty = min(rewarded_trajectory_statistics, key=rewarded_trajectory_statistics.get)
        obs, info = env.reset_by_module_and_difficulty(module_name, difficulty, train=train)
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
        logits_batch = model(obs_batch.cuda()).detach().cpu().numpy()
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


def reset_environment(env, rewarded_trajectory_statistics):
    module_name, difficulty = min(rewarded_trajectory_statistics, key=rewarded_trajectory_statistics.get)
    obs, info = env.reset_by_module_and_difficulty(module_name, difficulty)
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


# define and init environment
filenames = read_text_file("environment/module_lists/most_natural_composed_for_program_synthesis.txt").split("\n")
# TODO: undo hack to speedup experiments
# filepaths = [f"mathematics_dataset-v1.0/train-easy/algebra__linear_1d.txt"]
filepaths = [
    f"mathematics_dataset-v1.0/train-easy/{fn}" for fn in filenames if 'composed' not in fn
]
filepaths.remove('mathematics_dataset-v1.0/train-easy/algebra__linear_1d.txt')
filepaths.remove('mathematics_dataset-v1.0/train-easy/algebra__linear_2d.txt')
filepaths.remove('mathematics_dataset-v1.0/train-easy/algebra__polynomial_roots.txt')
env_config = {
    "problem_filepaths": filepaths,
    "corpus_filepath": str(Path("environment/corpus/10k_corpus.txt").resolve()),
    "num_problems_per_module": 10 ** 3,
    "validation_percentage": 0.2,
    "max_sequence_length": 500,
    "vocab_size": 200
}

# define search parameters
verbose = False
num_steps = 5000000
num_environments = 32
max_difficulty_level = 1

# initialize all environments
envs = init_envs(env_config, num_environments)
_, rewarded_trajectory_statistics = init_trajectory_data_structures(envs[0])

# architecture params
ntoken = env_config['vocab_size'] + 1
nhead = 4
nhid = 128
nlayers = 1
num_outputs = len(envs[0].actions)
dropout = 0.1

# training params
batch_size = 16
buffer_threshold = batch_size
positive_to_negative_ratio = 1
lr = 1
lr_decay_factor = 0.8
max_grad_norm = 0.5
n_required_validation_episodes = 100

# load model
# TODO: set load_model as param
load_model = False
if load_model:
    model = torch.load('modelling/models/model.pt')
else:
    dummy_model = None
    model = TransformerEncoderModel(ntoken=ntoken, nhead=nhead, nhid=nhid, nlayers=nlayers, num_outputs=num_outputs,
                                dropout=dropout)

model.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=lr_decay_factor)

# reset all environments
buffer = []
buffer_positives = 1
buffer_negatives = 1  # init to 1 to prevent division by zero
total_batches = 0
obs_batch, envs_info = reset_all(envs)
# take steps in all environments num_parallel_steps times
num_parallel_steps = num_steps // num_environments
for parallel_step_i in tqdm(range(num_parallel_steps)):
    # take a step in each environment in "parallel"
    action_batch = get_action_batch(obs_batch, envs, model=model)
    obs_batch, step_batch = step_all(envs, action_batch)
    # for each environment process the most recent step
    for env_i, ((obs, reward, done, info), action) in enumerate(zip(step_batch, action_batch)):
        envs_info[env_i]['trajectory'].append((obs.astype(np.int16), action, reward, done, info))
        # if episode is complete, check if trajectory should be kept in buffer and reset environment
        if done:
            update_rewarded_trajectory_statistics(envs_info[env_i], rewarded_trajectory_statistics)
            with open('modelling/training_graphs.txt', 'a') as f:
                f.write(f"{info['raw_observation']} = {envs[env_i].compute_graph.eval()}\n")
            if reward == 1 and verbose:
                print(f"{info['raw_observation']} = {envs[env_i].compute_graph.eval()}")
            if buffer_positives/buffer_negatives <= positive_to_negative_ratio and reward == 1:
                buffer_trajectory = extract_buffer_trajectory(envs_info[env_i]['trajectory'], reward)
                buffer.extend(buffer_trajectory)
                buffer_positives += 1
            elif buffer_positives/buffer_negatives > positive_to_negative_ratio and reward == -1:
                buffer_trajectory = extract_buffer_trajectory(envs_info[env_i]['trajectory'], reward)
                buffer.extend(buffer_trajectory)
                buffer_negatives += 1
            obs_batch[env_i], envs_info[env_i] = reset_environment(envs[env_i], rewarded_trajectory_statistics)
    # when enough steps have been put into buffer, construct training batches and fit
    if len(buffer) > buffer_threshold:
        # n_batches = math.floor(len(buffer) / batch_size)
        n_batches = 1
        total_batches += n_batches
        random.shuffle(buffer)
        for batch_i in range(n_batches):
            batch = buffer[batch_i * batch_size : (batch_i + 1) * batch_size]
            state_batch = torch.from_numpy(np.concatenate([np.expand_dims(step[0], 0) for step in batch]).astype(np.int64)).cuda()
            action_batch = torch.from_numpy(np.concatenate([np.expand_dims(step[1], 0) for step in batch]).astype(np.int64)).cuda()
            reward_batch = torch.from_numpy(np.concatenate([np.expand_dims(step[2], 0) for step in batch]).astype(np.int64)).cuda()
            batch_logits = model(state_batch)
            batch_probs = torch.softmax(batch_logits, axis=1)
            # loss is given by -mean(log(model(a=a_t|s_t)) * R_t)
            loss = -torch.mean(torch.log(batch_probs[:, action_batch]) * reward_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        # reset buffer
        buffer = []
        # inspect validation performance
        print(f'total_batches: {total_batches}')
        if total_batches % 10 == 0:
            scheduler.step()

        if total_batches % 1 == 0:
            total_reward = 0
            n_completed_validation_episodes = 0
            obs_batch, envs_info = reset_all(envs, train=False)
            # take steps in all environments num_parallel_steps times
            num_parallel_steps = num_steps // num_environments
            for parallel_step_i in range(num_parallel_steps):
                # take a step in each environment in "parallel"
                action_batch = get_action_batch(obs_batch, envs, model=model)
                obs_batch, step_batch = step_all(envs, action_batch)
                # for each environment process the most recent step
                for env_i, ((obs, reward, done, info), action) in enumerate(zip(step_batch, action_batch)):
                    envs_info[env_i]['trajectory'].append((obs.astype(np.int16), action, reward, done, info))
                    # if episode is complete, check if trajectory should be kept in buffer and reset environment
                    if done:
                        with open('modelling/validation_graphs.txt', 'a') as f:
                            f.write(f"{info['raw_observation']} = {envs[env_i].compute_graph.eval()}\n")
                        n_completed_validation_episodes += 1
                        total_reward += reward
                        obs_batch[env_i], envs_info[env_i] = reset_environment(envs[env_i],
                                                                               rewarded_trajectory_statistics)
                if n_completed_validation_episodes > n_required_validation_episodes:
                    break
            print(f'{total_batches} batches completed, mean validation reward: {total_reward/n_completed_validation_episodes}')
