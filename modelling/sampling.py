from environment.envs import MathEnv
import numpy as np
from pathlib import Path
from utils import read_text_file


def sample_masked_action_from_model(env, model, obs):
    policy_vector = model(obs)
    masked_policy_vector = env.mask_invalid_types(policy_vector)
    choices = np.arange(len(env.actions))
    action_index = np.random.choice(choices, p=masked_policy_vector)
    return action_index


def run_iteration(env, model, verbose=False):
    obs, _ = env.reset_with_same_problem()
    while not done:
        action_index = sample_masked_action_from_model(env, model, obs)
        obs, reward, done, info = env.step(action_index)
    return reward, env.compute_graph


filenames = read_text_file("environment/module_lists/most_natural_composed_for_program_synthesis.txt").split("\n")
filepaths = [
    f"mathematics_dataset-v1.0/train-easy/{fn}" for fn in filenames if 'composed' not in fn
]
env_config = {
    "problem_filepaths": filepaths,
    "corpus_filepath": str(Path("environment/corpus/10k_corpus.txt").resolve()),
    "num_problems_per_module": 10 ** 1,
    "validation_percentage": 0.2,
    "max_sequence_length": 100,
    "vocab_size": 200
}

n_iterations = 10 ** 8
max_attemps_per_problem = 10 ** 4
max_difficulty_level = 1

# retain correct and incorrect graphs
# TODO
correct_graphs = {}
incorrect_graphs = {}

env = MathEnv(env_config)

# track counts of correct graphs
correct_graph_statistics = {}
for module_name in env.train.keys():
    for difficulty in env.train[module_name].keys():
        if (module_name, difficulty) not in correct_graph_statistics and difficulty <= max_difficulty_level:
            correct_graph_statistics[(module_name,difficulty)] = 0


sample_new_problem = True
for i in range(n_iterations):
    if sample_new_problem:
        next_module, next_difficulty = min(correct_graph_statistics, key=correct_graph_statistics.get)
        env.reset_by_module_and_difficulty(next_module, next_difficulty)
        sample_new_problem = False
        attempts_to_guess_graph = 0
    reward, graph = run_iteration(env, model)
    if reward == 1:
        sample_new_problem = True
        correct_graphs[module_name][difficulty] = graph
        correct_graph_statistics[(module_name,difficulty)] += 1
    else:
        incorrect_graphs[module_name][difficulty] = graph
        attempts_to_guess_graph += 1
        if attempts_to_guess_graph > max_attemps_per_problem:
            sample_new_problem = True

