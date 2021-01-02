import copy
import torch
import numpy as np
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
        obs, _ = env.reset_by_module_and_difficulty(module_name, difficulty)
        envs_info.append({'trajectory': list(), 'module_name': module_name, 'difficulty': difficulty})
        obs_batch.append(obs)
    return obs_batch, envs_info


def get_obs_batch(envs, action_batch):
    return [env.step(action) for env, action in zip(envs, action_batch)]


def get_action_batch(model, obs_batch, envs):
    obs_batch = torch.from_numpy(np.array(obs_batch))
    logits_batch = model(obs_batch).detach().numpy()
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


# define and init environment
filenames = read_text_file("environment/module_lists/most_natural_composed_for_program_synthesis.txt").split("\n")
filepaths = [
    f"mathematics_dataset-v1.0/train-easy/{fn}" for fn in filenames if 'composed' not in fn
]
env_config = {
    "problem_filepaths": filepaths,
    "corpus_filepath": str(Path("environment/corpus/10k_corpus.txt").resolve()),
    "num_problems_per_module": 10 ** 3,
    "validation_percentage": 0.2,
    "max_sequence_length": 100,
    "vocab_size": 200
}
env = MathEnv(env_config)

# define search parameters
n_iterations = 10 ** 5
max_attemps_per_problem = 5*10 ** 3
max_difficulty_level = 1
rewarded_trajectories, rewarded_trajectory_statistics = init_rewarded_trajectories_data_structures(env)


# architecture params
ntoken = env_config['vocab_size'] + 1
nhead = 4
nhid = 128
nlayers = 1
num_outputs = len(env.actions)
dropout = 0.2

# load model
model = TransformerEncoderModel(ntoken=ntoken, nhead=nhead, nhid=nhid, nlayers=nlayers, num_outputs=num_outputs,
                                dropout=dropout)

num_steps = 100
num_environments = 10
# initialize and reset all environments
envs = init_envs(env_config, num_environments)
obs_batch, envs_info = reset_all(envs)
# take steps in all environments num_parallel_steps times
num_parallel_steps = num_steps // num_environments
for _ in range(num_parallel_steps):
    # take a step in each environment in "parallel"
    action_batch = get_action_batch(model, obs_batch, envs)
    step_batch = get_obs_batch(envs, action_batch)
    # for each environment process the most recent step
    for i, (obs, reward, done, info) in enumerate(step_batch):
        if done and reward == 1:
            # save the trajectory and reset the environment
            rewarded_trajectories[module_name][difficulty] = envs_info[i]['trajectory']
            rewarded_trajectory_statistics[(module_name, difficulty)] += 1
            module_name, difficulty = min(rewarded_trajectory_statistics, key=rewarded_trajectory_statistics.get)
            obs, _ = envs[i].reset_by_module_and_difficulty(module_name, difficulty)
            envs_info[i]['trajectory'] = []
            envs_info[i]['module_name'] = module_name
            envs_info[i]['difficulty'] = difficulty