from hparams import HParams
hparams = HParams('artifacts/.', hparams_filename='hparams_guessing', name='rl_math', ask_before_deletion=False)
from environment.envs.math_env import MathEnv
from environment.utils import guess_until_problem_solved
import unittest
from environment.utils import load_question_answer_pairs


class Test(unittest.TestCase):

    def test_guess_until_correct(self):
        """this test only terminates when the graph is correctly guessed or timeout is reached"""
        env = MathEnv(hparams.env)
        question_answer_pairs = load_question_answer_pairs('artifacts/short_problems.txt')
        for question, answer in question_answer_pairs:
            guess_until_problem_solved(env, question, answer, verbose=False, max_episode_index=50000)

