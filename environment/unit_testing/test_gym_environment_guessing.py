from environment.envs.math_env import MathEnv
import unittest


def guess_until_problem_solved(
    env, problem_index, verbose=False, max_episode_index=1000
):
    episode_i = 0
    graph_guessed_correctly = False
    encoded_problem_statement = env.reset_with_specific_problem(
        "short_problems", 1, problem_index
    )
    print(f"problem statement: {env.decode(encoded_problem_statement)}")
    while not graph_guessed_correctly and episode_i < max_episode_index:
        _ = env.reset_with_specific_problem("short_problems", 1, problem_index)
        done = False
        step_i = 0
        if verbose:
            print(f"episode: {episode_i}")
        while not done:
            action_index = env.sample_masked_action_index()
            observation, reward, done, info = env.step(action_index)
            # if verbose:
            #     print(f"\t\tS': {observation}, R: {reward}, done: {done}")
            if reward == 1:
                graph_guessed_correctly = True
            step_i += 1
        episode_i += 1
    print(f'graph: {info["raw_observation"].split(";")[1]}')
    print(f"trials taken to guess problem #{problem_index}: {episode_i}")


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
        for i in range(4,13):
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
