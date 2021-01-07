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
        encoded_problem_statement, _ = env.reset_with_specific_problem(
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
            == f"{problem_statement}; Eq('0 = 4*b + b + 15')"
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
        encoded_problem_statement, _ = env.reset_with_specific_problem(
            "short_problems", 1, 0
        )
        problem_statement = env.decode(encoded_problem_statement)
        assert problem_statement == "Solve 0 = 4*b + b + 15 for b."
        action = ss
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
            info["raw_observation"] == f"{problem_statement}; ss('param_0')"
        )
        assert reward == 0
        assert not done
        # assert that l_v & ap are the only actions not masked
        # policy_vector = env.sample_masked_policy_vector()
        # np.testing.assert_equal(np.ceil(policy_vector), np.array([1,0,1,1,0,0]))
        # next action
        action = "f0"
        action_index = env.get_action_index(action)
        observation_, reward, done, info = env.step(action_index)
        assert (
            info["raw_observation"]
            == f"{problem_statement}; ss(Eq('0 = 4*b + b + 15'))"
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
        encoded_problem_statement, _ = env.reset_with_specific_problem(
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
        encoded_problem_statement, _ = env.reset_with_specific_problem(
            "short_problems", 1, 0
        )
        problem_statement = env.decode(encoded_problem_statement)
        assert problem_statement == "Solve 0 = 4*b + b + 15 for b."
        action = l_v
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
            info["raw_observation"]
            == f"{problem_statement}; l_v('param_0','param_1')"
        )
        assert reward == 0
        assert not done
        assert env.compute_graph.current_node == env.compute_graph.root
        # assert that l_v & ss are the only actions not masked
        # because dict: is object, is dict, is not list, is not Eq, is not Var
        # policy_vector = env.sample_masked_policy_vector()
        # np.testing.assert_equal(np.ceil(policy_vector), np.array([1, 1, 0, 0, 0, 0]))
        # next action
        action = ss
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
            info["raw_observation"]
            == f"{problem_statement}; l_v(ss('param_0'),'param_1')"
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
            == f"{problem_statement}; l_v(ss('param_0'),Var('b'))"
        )
        assert reward == 0
        assert not done
        # current node is now the ss node because the l_v node has its args set
        assert env.compute_graph.current_node == env.compute_graph.root.args[0]
        # next action
        action = ape
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
            info["raw_observation"]
            == f"{problem_statement}; l_v(ss(ape('param_0')),Var('b'))"
        )
        assert reward == 0
        assert not done
        # next action
        action = "f0"
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
            info["raw_observation"]
            == f"{problem_statement}; l_v(ss(ape(Eq('0 = 4*b + b + 15'))),Var('b'))"
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
        encoded_problem_statement, _ = env.reset_with_specific_problem(
            "short_problems", 1, 5
        )
        problem_statement = env.decode(encoded_problem_statement)
        assert problem_statement == "Calculate the remainder when 93 is divided by 59."
        assert env.compute_graph.formal_elements == [Val("93"), Val("59")]
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
            == f"{problem_statement}; mod(Val('93'),'param_1')"
        )
        assert reward == 0
        assert not done
        # next action
        action = "f1"
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
            info["raw_observation"]
            == f"{problem_statement}; mod(Val('93'),Val('59'))"
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
        encoded_problem_statement, _ = env.reset_with_specific_problem(
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
                == f"{problem_statement}; gcd(Val('1300'),'param_1')"
        )
        assert reward == 0
        assert not done
        # next action
        action = "f1"
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
                info["raw_observation"]
                == f"{problem_statement}; gcd(Val('1300'),Val('300'))"
        )
        assert reward == 1
        assert done

    def test_mode_is_prime_success_1(self):
        env_config = {
            "problem_filepaths": ["artifacts/short_problems.txt"],
            "corpus_filepath": "../../environment/corpus/1k_corpus.txt",
            "num_problems_per_module": 10 ** 7,
            "validation_percentage": 0,
            "mode": "ip",
            "max_sequence_length": 100,
            "vocab_size": 200
        }
        env = MathEnv(env_config)
        # reset - then succeed after 4th action
        encoded_problem_statement, _ = env.reset_with_specific_problem(
            "short_problems", 1, 8
        )
        problem_statement = env.decode(encoded_problem_statement)
        assert problem_statement == "Is 93163 a prime number?"
        # first action
        action = ip
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
                info["raw_observation"] == f"{problem_statement}; ip('param_0')"
        )
        assert reward == 0
        assert not done
        # next action
        action = "f0"
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
                info["raw_observation"]
                == f"{problem_statement}; ip(Val('93163'))"
        )
        assert reward == 1
        assert done

    def test_mode_is_prime_success_2(self):
        env_config = {
            "problem_filepaths": ["artifacts/short_problems.txt"],
            "corpus_filepath": "../../environment/corpus/1k_corpus.txt",
            "num_problems_per_module": 10 ** 7,
            "validation_percentage": 0,
            "mode": "ip",
            "max_sequence_length": 100,
            "vocab_size": 200
        }
        env = MathEnv(env_config)
        # reset - then succeed after 4th action
        encoded_problem_statement, _ = env.reset_with_specific_problem(
            "short_problems", 1, 9
        )
        problem_statement = env.decode(encoded_problem_statement)
        assert problem_statement == "Is 66574 a composite number?"
        # first action
        action = nt
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
                info["raw_observation"] == f"{problem_statement}; nt('param_0')"
        )
        assert reward == 0
        assert not done
        # next action
        action = ip
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
                info["raw_observation"]
                == f"{problem_statement}; nt(ip('param_0'))"
        )
        assert reward == 0
        assert not done
        # next action
        action = "f0"
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
                info["raw_observation"]
                == f"{problem_statement}; nt(ip(Val('66574')))"
        )
        assert reward == 1
        assert done

    def test_not_op(self):
        env_config = {
            "problem_filepaths": ["artifacts/short_problems.txt"],
            "corpus_filepath": "../../environment/corpus/1k_corpus.txt",
            "num_problems_per_module": 10 ** 7,
            "validation_percentage": 0,
            "mode": "ip",
            "max_sequence_length": 100,
            "vocab_size": 200
        }
        env = MathEnv(env_config)
        # reset - then succeed after 4th action
        encoded_problem_statement, _ = env.reset_with_specific_problem(
            "short_problems", 1, 9
        )
        problem_statement = env.decode(encoded_problem_statement)
        assert problem_statement == "Is 66574 a composite number?"
        # take action
        action = nt
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
                info["raw_observation"] == f"{problem_statement}; nt('param_0')"
        )
        assert reward == 0
        assert not done
        # take action
        action = nt
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
                info["raw_observation"] == f"{problem_statement}; nt(nt('param_0'))"
        )
        assert reward == 0
        assert not done