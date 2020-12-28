from random import sample
from utils import read_text_file
from environment.envs.math_env import MathEnv
from tqdm import tqdm
from pathlib import Path

filenames = read_text_file("environment/module_lists/composed.txt").split("\n")
filepaths = [
    f"mathematics_dataset-v1.0/train-easy/{filename}" for filename in filenames
]
# env_config = {
#     "problem_filepaths": filepaths[:1],  # TODO: increase
#     "corpus_filepath": "environment/corpus/10k_corpus.txt",
#     "num_problems_per_module": 10 ** 5,
#     "p_val": 0,
# }

env_config = {
    "problem_filepaths": [
        str(Path("mathematics_dataset-v1.0/train-easy/numbers__gcd.txt").resolve())
    ],  # TODO hardcode single path to make this easy to run
    "corpus_filepath": str(Path("environment/corpus/10k_corpus.txt").resolve()),
    "num_problems_per_module": 10 ** 7,
    # data used for validation
    "p_val": 0.2,
}

env = MathEnv(env_config)
all_observations = []
# TODO: why does it get stuck
for _ in tqdm(range(int(10 ** 4))):
    done = False
    _ = env.reset()
    episode_observations = [env.problem_statement + '; ']
    while not done:
        # action = env.sample_masked_action_index()
        action = env.sample_action_index()
        observation, reward, done, info = env.step(action)
        episode_observations.append(info["raw_observation"])
        if reward == 1:
            break
    random_episode_observation = sample(episode_observations, 1)[0]
    all_observations.append(random_episode_observation)

all_observations = "\n".join(all_observations)
with open("environment/corpus/gcd_corpus.txt", "w") as f:
    f.write(all_observations)
