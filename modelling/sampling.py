import torch
from environment.envs import MathEnv
import numpy as np
from pathlib import Path
from utils import read_text_file
from scipy.special import softmax
from modelling.transformer_encoder import TransformerEncoderModel
from tqdm import tqdm

def sample_masked_action_from_model(env, model, obs):
    policy_vector = softmax(model(obs).detach().numpy()[0])
    masked_policy_vector = env.mask_invalid_types(policy_vector)
    masked_normed_policy_vector = masked_policy_vector / np.sum(
        masked_policy_vector
    )
    choices = np.arange(len(env.actions))
    action_index = np.random.choice(choices, p=masked_normed_policy_vector)
    return action_index


def run_iteration(env, model, verbose=False):
    obs, _ = env.reset_with_same_problem()
    trajectory = [(obs, None, None, None)]
    done = False
    while not done:
        obs = np.expand_dims(obs, 0)
        obs = torch.from_numpy(obs)
        # TEMP: commented out model inference to run this faster
        # action_index = sample_masked_action_from_model(env, model, obs)
        action_index = env.sample_masked_action_index()
        obs, reward, done, info = env.step(action_index)
        trajectory.append((obs, reward, done, info))
    return trajectory


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

# define data structures to track correct graphs
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

# sample graphs
sample_new_problem = True
for i in tqdm(range(n_iterations)):
    if sample_new_problem:
        module_name, difficulty = min(rewarded_trajectory_statistics, key=rewarded_trajectory_statistics.get)
        _, info = env.reset_by_module_and_difficulty(module_name, difficulty)
        sample_new_problem = False
        attempts_to_guess_graph = 0
    trajectory = run_iteration(env, model)
    final_reward = trajectory[-1][1]
    if final_reward == 1:
        print(trajectory[-1][3]['raw_observation'])
        sample_new_problem = True
        rewarded_trajectories[module_name][difficulty] = trajectory
        rewarded_trajectory_statistics[(module_name,difficulty)] += 1
    else:
        attempts_to_guess_graph += 1
        if attempts_to_guess_graph > max_attemps_per_problem:
            sample_new_problem = True

