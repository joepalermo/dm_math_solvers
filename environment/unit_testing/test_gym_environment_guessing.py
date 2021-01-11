from hparams import HParams
hparams = HParams('artifacts/.', hparams_filename='hparams', name='rl_math', ask_before_deletion=False)
from environment.envs.math_env import MathEnv
from environment.utils import guess_until_problem_solved
import unittest


class Test(unittest.TestCase):

    def test_guess_until_correct(self):
        """this test only terminates when the graph is correctly guessed or timeout is reached"""
        env_config = {
            "problem_filepaths": ["artifacts/short_problems.txt"],
            "corpus_filepath": "../../environment/corpus/1k_corpus.txt",
            "num_problems_per_module": 10 ** 7,
            "validation_percentage": 0,
            "max_sequence_length": 100,
            "vocab_size": 200
        }
        env = MathEnv(env_config)
        for i in range(0,12):
            guess_until_problem_solved(env, i, verbose=False, max_episode_index=50000)

    # def test(self):
    #     env_config = {
    #         "problem_filepaths": ["artifacts/short_problems.txt"],
    #         "corpus_filepath": "../../environment/corpus/1k_corpus.txt",
    #         "num_problems_per_module": 10 ** 7,
    #         "validation_percentage": 0,
    #         "gcd_test": False
    #     }
    #     env = MathEnv(env_config)
    #     print(env.observation_space)
    #
    #
    # def test_load_all_problems(self):
    #     filenames = read_text_file('../module_lists/composed.txt').split('\n')
    #     filepaths = [f'../../mathematics_dataset-v1.0/train-easy/{filename}' for filename in filenames]
    #     env_config = {'problem_filepaths': filepaths,
    #                   'corpus_filepath': '../../environment/corpus/1k_corpus.txt',
    #                   'num_problems_per_module': 10 ** 7,
    #                   'p_val': 0}
    #     env = MathEnv(env_config)
    #     print()
