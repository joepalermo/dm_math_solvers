from hparams import HParams
hparams = HParams('.', hparams_filename='hparams', name='rl_math', ask_before_deletion=False)
from environment.envs.math_env import MathEnv
from random import sample
from tqdm import tqdm

num_problems = 50000
env = MathEnv(hparams.env)

all_observations = []
for i in range(num_problems):
    print(i)
    done = False
    episode_observations = [env.reset(train=False)[1]['raw_observation']]
    # TODO: ensure that questions from question_corpus aren't used
    while not done:
        action = env.sample_masked_action_index()
        observation, reward, done, info = env.step(action)
        episode_observations.append(info["raw_observation"])
    random_episode_observation = sample(episode_observations, 1)[0]
    all_observations.append(random_episode_observation)

all_observations = "\n".join(all_observations)
with open("environment/tokenization/graph_corpus.txt", "w") as f:
    f.write(all_observations)

