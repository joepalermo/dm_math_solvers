from hparams import HParams
hparams = HParams('artifacts/.', hparams_filename='hparams', name='rl_math', ask_before_deletion=False)
from environment.envs.math_env import MathEnv
from environment.utils import guess_until_problem_solved
import unittest


class Test(unittest.TestCase):

    def test_guess_until_correct(self):
        """this test only terminates when the graph is correctly guessed or timeout is reached"""
        env = MathEnv(hparams.env)
        for i in range(0,12):
            guess_until_problem_solved(env, i, verbose=False, max_episode_index=50000)

