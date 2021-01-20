from hparams import HParams
hparams = HParams('.', hparams_filename='hparams', name='rl_math', ask_before_deletion=False)
from environment.envs.math_env import MathEnv
from random import shuffle
from tqdm import tqdm

num_problems = 50000
env = MathEnv(hparams.env)

questions = []
for _ in tqdm(range(num_problems)):
    question = env.reset(train=True)[1]['raw_observation'] + ';'  # append semicolon delimiter
    questions.append(question)

shuffle(questions)
with open("environment/tokenization/question_corpus.txt", "w") as f:
    f.write("\n".join(questions))
