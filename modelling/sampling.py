import copy
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from utils import read_text_file
from scipy.special import softmax
from environment.envs import MathEnv
from modelling.transformer_encoder import TransformerEncoderModel


def init_rewarded_trajectories_data_structures(env):
    '''define data structures to track correct graphs'''
    rewarded_trajectories = {}
    rewarded_trajectory_statistics = {}
    for module_name in env.train.keys():
        if module_name not in rewarded_trajectories:
            rewarded_trajectories[module_name] = {}
        for difficulty in env.train[module_name].keys():
            if difficulty not in rewarded_trajectories[module_name]:
                rewarded_trajectories[module_name][difficulty] = []
            if (module_name, difficulty) not in rewarded_trajectory_statistics and difficulty <= max_difficulty_level:
                rewarded_trajectory_statistics[(module_name, difficulty)] = 0
    return rewarded_trajectories, rewarded_trajectory_statistics


def init_envs(env_config, num_environments=10):
    env = MathEnv(env_config)
    envs = [env]
    envs.extend([copy.copy(env) for _ in range(1, num_environments)])
    return envs


def reset_all(envs):
    envs_info = []
    obs_batch = []
    for env in envs:
        module_name, difficulty = min(rewarded_trajectory_statistics, key=rewarded_trajectory_statistics.get)
        obs, info = env.reset_by_module_and_difficulty(module_name, difficulty)
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
        obs_batch = torch.from_numpy(obs_batch)
        logits_batch = model(obs_batch).detach().numpy()
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


def save_rewarded_trajectory(env_info, rewarded_trajectories, rewarded_trajectory_statistics):
    module_name = env_info['module_name']
    difficulty = env_info['difficulty']
    trajectory = env_info['trajectory']
    rewarded_trajectories[module_name][difficulty] = trajectory
    rewarded_trajectory_statistics[(module_name, difficulty)] += 1


def reset_environment(env, rewarded_trajectory_statistics):
    module_name, difficulty = min(rewarded_trajectory_statistics, key=rewarded_trajectory_statistics.get)
    obs, info = env.reset_by_module_and_difficulty(module_name, difficulty)
    return obs, {'problem_statement': info['raw_observation'],
                 'trajectory': [(obs, None, None, None)],
                 'module_name': module_name,
                 'difficulty': difficulty}


# define and init environment
filenames = read_text_file("environment/module_lists/most_natural_composed_for_program_synthesis.txt").split("\n")
filepaths = [
    f"mathematics_dataset-v1.0/train-easy/{fn}" for fn in filenames if 'composed' not in fn
]
# TODO: undo hack to speedup experiments
filepaths.remove('mathematics_dataset-v1.0/train-easy/algebra__linear_1d.txt')
filepaths.remove('mathematics_dataset-v1.0/train-easy/algebra__linear_2d.txt')
filepaths.remove('mathematics_dataset-v1.0/train-easy/algebra__polynomial_roots.txt')
env_config = {
    "problem_filepaths": filepaths,
    "corpus_filepath": str(Path("environment/corpus/10k_corpus.txt").resolve()),
    "num_problems_per_module": 10 ** 3,
    "validation_percentage": 0.2,
    "max_sequence_length": 400,
    "vocab_size": 200
}

# define search parameters
num_steps = 1000000
num_environments = 32
max_difficulty_level = 1

# initialize all environments
envs = init_envs(env_config, num_environments)
rewarded_trajectories, rewarded_trajectory_statistics = init_rewarded_trajectories_data_structures(envs[0])

# architecture params
ntoken = env_config['vocab_size'] + 1
nhead = 4
nhid = 128
nlayers = 1
num_outputs = len(envs[0].actions)
dropout = 0.2

# load model
# TODO: set load_model as param
load_model = False
if load_model:
    model = torch.load('modelling/models/model.pt')
else:
    model = None
    # model = TransformerEncoderModel(ntoken=ntoken, nhead=nhead, nhid=nhid, nlayers=nlayers, num_outputs=num_outputs,
    #                             dropout=dropout)

# reset all environments
obs_batch, envs_info = reset_all(envs)
# take steps in all environments num_parallel_steps times
num_parallel_steps = num_steps // num_environments
for _ in tqdm(range(num_parallel_steps)):
    # take a step in each environment in "parallel"
    action_batch = get_action_batch(obs_batch, envs, model=model)
    obs_batch, step_batch = step_all(envs, action_batch)
    # for each environment process the most recent step
    for i, (obs, reward, done, info) in enumerate(step_batch):
        envs_info[i]['trajectory'].append((obs, reward, done, info))
        if done:
            if reward == 1:
                save_rewarded_trajectory(envs_info[i], rewarded_trajectories, rewarded_trajectory_statistics)
                print(envs_info[i]['trajectory'][-1][3]['raw_observation'], envs[i].compute_graph.eval())
            obs_batch[i], envs_info[i] = reset_environment(envs[i], rewarded_trajectory_statistics)