from random import sample
from utils import read_text_file
from environment.envs.math_env import MathEnv
from tqdm import tqdm

filenames = read_text_file('../environment/module_lists/composed.txt').split('\n')
filepaths = [f'mathematics_dataset-v1.0/train-easy/{filename}' for filename in filenames]
env = MathEnv(filepaths, num_problems_per_module=int(1e5))
all_observations = []
# TODO: why does it get stuck
for _ in tqdm(range(int(1e3))):
    done = False
    episode_observations = [env.reset()]
    while not done:
        action = env.sample_masked_action()
        observation, reward, done, _ = env.step(action)
        episode_observations.append(observation)
        if reward == 1:
            break
    random_episode_observation = sample(episode_observations, 1)[0]
    all_observations.append(random_episode_observation)

all_observations = '\n'.join(all_observations)
with open('corpus/corpus.txt', 'w') as f:
    f.write(all_observations)
