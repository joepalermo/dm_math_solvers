from hparams import HParams
hparams = HParams('.', hparams_filename='hparams', name='rl_math', ask_before_deletion=False)
from environment.envs.math_env import MathEnv
from random import sample
from utils import read_text_file

question_corpus = read_text_file("environment/tokenization/question_corpus.txt")
question_set = set(question_corpus.split('\n'))

target_corpus_size = 25000
env = MathEnv(hparams.env)

all_observations = []
observations_added = 0
while observations_added < target_corpus_size:
    print(observations_added)
    question = env.reset(train=False)[1]['raw_observation'] + ';'
    # if the question is already in the question_corpus, then move on to another question
    if question in question_set:
        continue
    episode_observations = [question]
    done = False
    while not done:
        action = env.sample_masked_action_index()
        observation, reward, done, info = env.step(action)
        episode_observations.append(info["raw_observation"])
    random_episode_observation = sample(episode_observations, 1)[0]
    all_observations.append(random_episode_observation)
    observations_added += 1

all_observations = "\n".join(all_observations)
with open("environment/tokenization/graph_corpus.txt", "w") as f:
    f.write(all_observations)

