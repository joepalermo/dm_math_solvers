import numpy as np
import gym
from environment.utils import extract_formal_elements
from environment.envs.math_env import MathEnv
from environment.typed_operators import *
from utils import read_text_file
import unittest
import glob
import os


def guess_until_problem_solved(env, problem_index, verbose=False, max_episode_index=1000):
    episode_i = 0
    graph_guessed_correctly = False
    print(f'problem statement: {env.reset_with_specific_problem("short_problems", 1, problem_index)}')
    while not graph_guessed_correctly and episode_i < max_episode_index:
        _ = env.reset_with_specific_problem('short_problems', 1, problem_index)
        done = False
        step_i = 0
        if verbose:
            print(f"episode: {episode_i}")
        while not done:
            action = env.sample_masked_action()
            observation, reward, done, info = env.step(action)
            # if verbose:
            #     print(f"\t\tS': {observation}, R: {reward}, done: {done}")
            if reward == 1:
                graph_guessed_correctly = True
            step_i += 1
        episode_i += 1
    print(f'graph: {info["raw_observation"].split(";")[1]}')
    print(f'trials taken to guess problem #{problem_index}: {episode_i}')


class Test(unittest.TestCase):

    def test_problem_0_fail_1(self):
        env_config = {'problem_filepaths': ['artifacts/short_problems.txt'],
                      'corpus_filepath': '../../environment/corpus/minimal_corpus.txt',
                      'num_problems_per_module': 10**7,
                      'p_val': 0}
        env = MathEnv(env_config)
        # reset - then fail after 1st action
        problem_statement = env.reset_with_specific_problem('short_problems', 1, 0)
        f = extract_formal_elements(problem_statement)  # for use below
        assert f == ['0 = 4*b + b + 15', 'b']
        action = 'f0'
        formal_element = env.compute_graph.lookup_formal_element(action)
        observation_, reward, done, info = env.step(action)
        assert info['raw_observation'] == f"{problem_statement}; Equation('0 = 4*b + b + 15')"
        assert reward == 0
        assert done

    def test_problem_0_fail_2(self):
        env_config = {'problem_filepaths': ['artifacts/short_problems.txt'],
                      'corpus_filepath': '../../environment/corpus/minimal_corpus.txt',
                      'num_problems_per_module': 10 ** 7,
                      'p_val': 0}
        env = MathEnv(env_config)
        # reset - then fail after 2nd action
        problem_statement = env.reset_with_specific_problem('short_problems', 1, 0)
        assert problem_statement == 'Solve 0 = 4*b + b + 15 for b.'
        action = solve_system
        observation, reward, done, info = env.step(action)
        assert info['raw_observation'] == f"{problem_statement}; solve_system('param_0')"
        assert reward == 0
        assert not done
        # assert that lookup_value & append are the only actions not masked
        # policy_vector = env.sample_masked_policy_vector()
        # np.testing.assert_equal(np.ceil(policy_vector), np.array([1,0,1,1,0,0]))
        # next action
        action = 'f0'
        observation_, reward, done, info = env.step(action)
        assert info['raw_observation'] == f"{problem_statement}; solve_system(Equation('0 = 4*b + b + 15'))"
        assert reward == 0
        assert done

    def test_problem_0_success_1(self):
        env_config = {'problem_filepaths': ['artifacts/short_problems.txt'],
                      'corpus_filepath': '../../environment/corpus/minimal_corpus.txt',
                      'num_problems_per_module': 10**7,
                      'p_val': 0}
        env = MathEnv(env_config)
        # reset - then succeed after 4th action
        problem_statement = env.reset_with_specific_problem('short_problems', 1, 0)
        assert problem_statement == 'Solve 0 = 4*b + b + 15 for b.'
        action = lookup_value
        observation, reward, done, info = env.step(action)
        assert info['raw_observation'] == f"{problem_statement}; lookup_value('param_0','param_1')"
        assert reward == 0
        assert not done
        assert env.compute_graph.current_node == env.compute_graph.root
        # assert that lookup_value & solve_system are the only actions not masked
        # because dict: is object, is dict, is not list, is not Equation, is not Variable
        # policy_vector = env.sample_masked_policy_vector()
        # np.testing.assert_equal(np.ceil(policy_vector), np.array([1, 1, 0, 0, 0, 0]))
        # next action
        action = solve_system
        observation, reward, done, info = env.step(action)
        assert info['raw_observation'] == f"{problem_statement}; lookup_value(solve_system('param_0'),'param_1')"
        assert reward == 0
        assert not done
        # current node is still root because it takes 2 arguments and only 1 has been given
        assert env.compute_graph.current_node == env.compute_graph.root
        # next action
        action = 'f1'
        observation, reward, done, info = env.step(action)
        assert info['raw_observation'] == f"{problem_statement}; lookup_value(solve_system('param_0'),Variable('b'))"
        assert reward == 0
        assert not done
        # current node is now the solve_system node because the lookup_value node has its args set
        assert env.compute_graph.current_node == env.compute_graph.root.args[0]
        #next action
        action = append_to_empty_list
        observation, reward, done, info = env.step(action)
        assert info['raw_observation'] == f"{problem_statement}; lookup_value(solve_system(append_to_empty_list('param_0')),Variable('b'))"
        assert reward == 0
        assert not done
        # next action
        action = 'f0'
        observation, reward, done, info = env.step(action)
        assert info['raw_observation'] == f"{problem_statement}; lookup_value(solve_system(append_to_empty_list(Equation('0 = 4*b + b + 15'))),Variable('b'))"
        assert reward == 1
        assert done

    def test_problem_5_success(self):
        env_config = {'problem_filepaths': ['artifacts/short_problems.txt'],
                      'corpus_filepath': '../../environment/corpus/minimal_corpus.txt',
                      'num_problems_per_module': 10 ** 7,
                      'p_val': 0}
        env = MathEnv(env_config)
        # reset - then succeed after 4th action
        problem_statement = env.reset_with_specific_problem('short_problems', 1, 5)
        assert problem_statement == 'Calculate the remainder when 93 is divided by 59.'
        assert env.compute_graph.formal_elements == [Value('93'), Value('59')]
        # first action
        action = mod
        observation, reward, done, info = env.step(action)
        assert info['raw_observation'] == f"{problem_statement}; mod('param_0','param_1')"
        assert reward == 0
        assert not done
        # next action
        action = 'f0'
        observation, reward, done, info = env.step(action)
        assert info['raw_observation'] == f"{problem_statement}; mod(Value('93'),'param_1')"
        assert reward == 0
        assert not done
        # next action
        action = 'f1'
        observation, reward, done, info = env.step(action)
        assert info['raw_observation'] == f"{problem_statement}; mod(Value('93'),Value('59'))"
        assert reward == 1
        assert done


    def test_guess_until_correct(self):
        '''this test only terminates when the graph is correctly guessed or timeout is reached'''
        env_config = {'problem_filepaths': ['artifacts/short_problems.txt'],
                      'corpus_filepath': '../../environment/corpus/minimal_corpus.txt',
                      'num_problems_per_module': 10 ** 7,
                      'p_val': 0}
        env = MathEnv(env_config)
        for i in range(4,10):
            guess_until_problem_solved(env, i, verbose=False, max_episode_index=50000)

    #
    # def test_load_all_problems(self):
    #     filenames = read_text_file('../module_lists/composed.txt').split('\n')
    #     filepaths = [f'../../mathematics_dataset-v1.0/train-easy/{filename}' for filename in filenames]
    #     env_config = {'problem_filepaths': filepaths,
    #                   'corpus_filepath': '../../environment/corpus/minimal_corpus.txt',
    #                   'num_problems_per_module': 10 ** 7,
    #                   'p_val': 0}
    #     env = MathEnv(env_config)
    #     print()

