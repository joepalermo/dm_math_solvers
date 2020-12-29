import numpy as np
import gym
from environment.utils import extract_formal_elements
from environment.envs.math_env import MathEnv
from environment.typed_operators import *
from utils import read_text_file
import unittest
import glob
import os


class Test(unittest.TestCase):
    def test_problem_0_fail_1(self):
        env_config = {
            "problem_filepaths": ["artifacts/short_problems.txt"],
            "corpus_filepath": "../../environment/corpus/1k_corpus.txt",
            "num_problems_per_module": 10 ** 7,
            "validation_percentage": 0,
            "max_sequence_length": 100,
            "vocab_size": 200
        }
        env = MathEnv(env_config)
        # reset - then fail after 1st action
        encoded_problem_statement = env.reset_with_specific_problem(
            "short_problems", 1, 0
        )
        problem_statement = env.decode(encoded_problem_statement)
        f = extract_formal_elements(problem_statement)  # for use below
        assert f == ["0 = 4*b + b + 15", "b"]
        action = "f0"
        action_index = env.get_action_index(action)
        observation_, reward, done, info = env.step(action_index)
        assert (
            info["raw_observation"]
            == f"{problem_statement}; Equation('0 = 4*b + b + 15')"
        )
        assert reward == 0
        assert done

    def test_problem_0_fail_2(self):
        env_config = {
            "problem_filepaths": ["artifacts/short_problems.txt"],
            "corpus_filepath": "../../environment/corpus/1k_corpus.txt",
            "num_problems_per_module": 10 ** 7,
            "validation_percentage": 0,
            "max_sequence_length": 100,
            "vocab_size": 200
        }
        env = MathEnv(env_config)
        # reset - then fail after 2nd action
        encoded_problem_statement = env.reset_with_specific_problem(
            "short_problems", 1, 0
        )
        problem_statement = env.decode(encoded_problem_statement)
        assert problem_statement == "Solve 0 = 4*b + b + 15 for b."
        action = solve_system
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
            info["raw_observation"] == f"{problem_statement}; solve_system('param_0')"
        )
        assert reward == 0
        assert not done
        # assert that lookup_value & append are the only actions not masked
        # policy_vector = env.sample_masked_policy_vector()
        # np.testing.assert_equal(np.ceil(policy_vector), np.array([1,0,1,1,0,0]))
        # next action
        action = "f0"
        action_index = env.get_action_index(action)
        observation_, reward, done, info = env.step(action_index)
        assert (
            info["raw_observation"]
            == f"{problem_statement}; solve_system(Equation('0 = 4*b + b + 15'))"
        )
        assert reward == 0
        assert done

    def test_problem_0_fail_3(self):
        env_config = {
            "problem_filepaths": ["artifacts/short_problems.txt"],
            "corpus_filepath": "../../environment/corpus/1k_corpus.txt",
            "num_problems_per_module": 10 ** 7,
            "validation_percentage": 0,
            "max_sequence_length": 100,
            "vocab_size": 200
        }
        env = MathEnv(env_config)
        # reset - then fail after 1st action
        encoded_problem_statement = env.reset_with_specific_problem(
            "short_problems", 1, 0
        )
        problem_statement = env.decode(encoded_problem_statement)
        f = extract_formal_elements(problem_statement)  # for use below
        assert f == ["0 = 4*b + b + 15", "b"]
        action = "f10"  # indexing out of range
        action_index = env.get_action_index(action)
        observation_, reward, done, info = env.step(action_index)
        assert reward == 0
        assert done

    def test_problem_0_success_1(self):
        env_config = {
            "problem_filepaths": ["artifacts/short_problems.txt"],
            "corpus_filepath": "../../environment/corpus/1k_corpus.txt",
            "num_problems_per_module": 10 ** 7,
            "validation_percentage": 0,
            "max_sequence_length": 100,
            "vocab_size": 200
        }
        env = MathEnv(env_config)
        # reset - then succeed after 4th action
        encoded_problem_statement = env.reset_with_specific_problem(
            "short_problems", 1, 0
        )
        problem_statement = env.decode(encoded_problem_statement)
        assert problem_statement == "Solve 0 = 4*b + b + 15 for b."
        action = lookup_value
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
            info["raw_observation"]
            == f"{problem_statement}; lookup_value('param_0','param_1')"
        )
        assert reward == 0
        assert not done
        assert env.compute_graph.current_node == env.compute_graph.root
        # assert that lookup_value & solve_system are the only actions not masked
        # because dict: is object, is dict, is not list, is not Equation, is not Variable
        # policy_vector = env.sample_masked_policy_vector()
        # np.testing.assert_equal(np.ceil(policy_vector), np.array([1, 1, 0, 0, 0, 0]))
        # next action
        action = solve_system
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
            info["raw_observation"]
            == f"{problem_statement}; lookup_value(solve_system('param_0'),'param_1')"
        )
        assert reward == 0
        assert not done
        # current node is still root because it takes 2 arguments and only 1 has been given
        assert env.compute_graph.current_node == env.compute_graph.root
        # next action
        action = "f1"
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
            info["raw_observation"]
            == f"{problem_statement}; lookup_value(solve_system('param_0'),Variable('b'))"
        )
        assert reward == 0
        assert not done
        # current node is now the solve_system node because the lookup_value node has its args set
        assert env.compute_graph.current_node == env.compute_graph.root.args[0]
        # next action
        action = append_to_empty_list
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
            info["raw_observation"]
            == f"{problem_statement}; lookup_value(solve_system(append_to_empty_list('param_0')),Variable('b'))"
        )
        assert reward == 0
        assert not done
        # next action
        action = "f0"
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
            info["raw_observation"]
            == f"{problem_statement}; lookup_value(solve_system(append_to_empty_list(Equation('0 = 4*b + b + 15'))),Variable('b'))"
        )
        assert reward == 1
        assert done

    def test_problem_5_success(self):
        env_config = {
            "problem_filepaths": ["artifacts/short_problems.txt"],
            "corpus_filepath": "../../environment/corpus/1k_corpus.txt",
            "num_problems_per_module": 10 ** 7,
            "validation_percentage": 0,
            "max_sequence_length": 100,
            "vocab_size": 200
        }
        env = MathEnv(env_config)
        # reset - then succeed after 4th action
        encoded_problem_statement = env.reset_with_specific_problem(
            "short_problems", 1, 5
        )
        problem_statement = env.decode(encoded_problem_statement)
        assert problem_statement == "Calculate the remainder when 93 is divided by 59."
        assert env.compute_graph.formal_elements == [Value("93"), Value("59")]
        # first action
        action = mod
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
            info["raw_observation"] == f"{problem_statement}; mod('param_0','param_1')"
        )
        assert reward == 0
        assert not done
        # next action
        action = "f0"
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
            info["raw_observation"]
            == f"{problem_statement}; mod(Value('93'),'param_1')"
        )
        assert reward == 0
        assert not done
        # next action
        action = "f1"
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
            info["raw_observation"]
            == f"{problem_statement}; mod(Value('93'),Value('59'))"
        )
        assert reward == 1
        assert done

    def test_problem_6_success(self):
        env_config = {
            "problem_filepaths": ["artifacts/short_problems.txt"],
            "corpus_filepath": "../../environment/corpus/1k_corpus.txt",
            "num_problems_per_module": 10 ** 7,
            "validation_percentage": 0,
            "gcd_test": True,
            "max_sequence_length": 100,
            "vocab_size": 200
        }
        env = MathEnv(env_config)
        # reset - then succeed after 4th action
        encoded_problem_statement = env.reset_with_specific_problem(
            "short_problems", 1, 6
        )
        problem_statement = env.decode(encoded_problem_statement)
        assert problem_statement == "Calculate the highest common divisor of 1300 and 300."
        # first action
        action = gcd
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
                info["raw_observation"] == f"{problem_statement}; gcd('param_0','param_1')"
        )
        assert reward == 0
        assert not done
        # next action
        action = "f0"
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
                info["raw_observation"]
                == f"{problem_statement}; gcd(Value('1300'),'param_1')"
        )
        assert reward == 0
        assert not done
        # next action
        action = "f1"
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
                info["raw_observation"]
                == f"{problem_statement}; gcd(Value('1300'),Value('300'))"
        )
        assert reward == 1
        assert done

    def test_mode_is_prime_success_1(self):
        env_config = {
            "problem_filepaths": ["artifacts/short_problems.txt"],
            "corpus_filepath": "../../environment/corpus/1k_corpus.txt",
            "num_problems_per_module": 10 ** 7,
            "validation_percentage": 0,
            "mode": "is_prime",
            "max_sequence_length": 100,
            "vocab_size": 200
        }
        env = MathEnv(env_config)
        # reset - then succeed after 4th action
        encoded_problem_statement = env.reset_with_specific_problem(
            "short_problems", 1, 8
        )
        problem_statement = env.decode(encoded_problem_statement)
        assert problem_statement == "Is 93163 a prime number?"
        # first action
        action = is_prime
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
                info["raw_observation"] == f"{problem_statement}; is_prime('param_0')"
        )
        assert reward == 0
        assert not done
        # next action
        action = "f0"
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
                info["raw_observation"]
                == f"{problem_statement}; is_prime(Value('93163'))"
        )
        assert reward == 1
        assert done

    def test_mode_is_prime_success_2(self):
        env_config = {
            "problem_filepaths": ["artifacts/short_problems.txt"],
            "corpus_filepath": "../../environment/corpus/1k_corpus.txt",
            "num_problems_per_module": 10 ** 7,
            "validation_percentage": 0,
            "mode": "is_prime",
            "max_sequence_length": 100,
            "vocab_size": 200
        }
        env = MathEnv(env_config)
        # reset - then succeed after 4th action
        encoded_problem_statement = env.reset_with_specific_problem(
            "short_problems", 1, 9
        )
        problem_statement = env.decode(encoded_problem_statement)
        assert problem_statement == "Is 66574 a composite number?"
        # first action
        action = not_op
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
                info["raw_observation"] == f"{problem_statement}; not_op('param_0')"
        )
        assert reward == 0
        assert not done
        # next action
        action = is_prime
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
                info["raw_observation"]
                == f"{problem_statement}; not_op(is_prime('param_0'))"
        )
        assert reward == 0
        assert not done
        # next action
        action = "f0"
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
                info["raw_observation"]
                == f"{problem_statement}; not_op(is_prime(Value('66574')))"
        )
        assert reward == 1
        assert done

    def test_not_op(self):
        env_config = {
            "problem_filepaths": ["artifacts/short_problems.txt"],
            "corpus_filepath": "../../environment/corpus/1k_corpus.txt",
            "num_problems_per_module": 10 ** 7,
            "validation_percentage": 0,
            "mode": "is_prime",
            "max_sequence_length": 100,
            "vocab_size": 200
        }
        env = MathEnv(env_config)
        # reset - then succeed after 4th action
        encoded_problem_statement = env.reset_with_specific_problem(
            "short_problems", 1, 9
        )
        problem_statement = env.decode(encoded_problem_statement)
        assert problem_statement == "Is 66574 a composite number?"
        # take action
        action = not_op
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
                info["raw_observation"] == f"{problem_statement}; not_op('param_0')"
        )
        assert reward == 0
        assert not done
        # take action
        action = not_op
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
                info["raw_observation"] == f"{problem_statement}; not_op(not_op('param_0'))"
        )
        assert reward == 0
        assert not done