from hparams import HParams
hparams = HParams('artifacts/.', hparams_filename='hparams', name='rl_math', ask_before_deletion=False)
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
            "short_problems", 0, 0
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
        assert reward == -1
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
            "short_problems", 0, 0
        )
        problem_statement = env.decode(encoded_problem_statement)
        assert problem_statement == "Solve 0 = 4*b + b + 15 for b."
        action = ss
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
            info["raw_observation"] == f"{problem_statement}; ss('p_0')"
        )
        assert reward == 0
        assert not done
        # assert that lv & ap are the only actions not masked
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
        assert reward == -1
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
            "short_problems", 0, 0
        )
        problem_statement = env.decode(encoded_problem_statement)
        f = extract_formal_elements(problem_statement)  # for use below
        assert f == ["0 = 4*b + b + 15", "b"]
        action = "f10"  # indexing out of range
        action_index = env.get_action_index(action)
        observation_, reward, done, info = env.step(action_index)
        assert reward == -1
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
            "short_problems", 0, 0
        )
        problem_statement = env.decode(encoded_problem_statement)
        assert problem_statement == "Solve 0 = 4*b + b + 15 for b."
        action = lv
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
            info["raw_observation"]
            == f"{problem_statement}; lv('p_0','p_1')"
        )
        assert reward == 0
        assert not done
        assert env.compute_graph.current_node == env.compute_graph.root
        # assert that lv & ss are the only actions not masked
        # because dict: is object, is dict, is not list, is not Eq, is not Var
        # policy_vector = env.sample_masked_policy_vector()
        # np.testing.assert_equal(np.ceil(policy_vector), np.array([1, 1, 0, 0, 0, 0]))
        # next action
        action = ss
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
            info["raw_observation"]
            == f"{problem_statement}; lv(ss('p_0'),'p_1')"
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
            == f"{problem_statement}; lv(ss('p_0'),Var('b'))"
        )
        assert reward == 0
        assert not done
        # current node is now the ss node because the lv node has its args set
        assert env.compute_graph.current_node == env.compute_graph.root.args[0]
        # next action
        action = ape
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
            info["raw_observation"]
            == f"{problem_statement}; lv(ss(ape('p_0')),Var('b'))"
        )
        assert reward == 0
        assert not done
        # next action
        action = "f0"
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
            info["raw_observation"]
            == f"{problem_statement}; lv(ss(ape(Eq('0 = 4*b + b + 15'))),Var('b'))"
        )
        assert reward == 1
        assert done

    def test_problem_4_success_1_with_masking(self):
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
            "short_problems", 0, 4
        )
        problem_statement = env.decode(encoded_problem_statement)
        assert problem_statement == "Find the first derivative of 2*d**4 - 35*d**2 - 695 wrt d."
        # take action
        action = dfw
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
                info["raw_observation"] == f"{problem_statement}; dfw('p_0','p_1')"
        )
        # take action
        action = "f0"
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert reward == 0
        assert not done
        assert (
                info["raw_observation"] == f"{problem_statement}; dfw(Ex('2*d**4 - 35*d**2 - 695'),'p_1')"
        )
        vector = np.ones(len(env.actions))
        masked_vector = env.mask_invalid_types(vector)
        assert masked_vector[env.get_action_index("f0")] == 0 and \
               masked_vector[env.get_action_index("f1")] == 1
        # take action
        action = "f1"
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert reward == 1
        assert done

    def test_problem_4_success_2_with_masking(self):
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
            "short_problems", 0, 4
        )
        problem_statement = env.decode(encoded_problem_statement)
        assert problem_statement == "Find the first derivative of 2*d**4 - 35*d**2 - 695 wrt d."
        # take action
        action = df
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
                info["raw_observation"] == f"{problem_statement}; df('p_0')"
        )
        # take action
        action = "f0"
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert reward == 1
        assert done
        assert (
                info["raw_observation"] == f"{problem_statement}; df(Ex('2*d**4 - 35*d**2 - 695'))"
        )

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
            "short_problems", 0, 5
        )
        problem_statement = env.decode(encoded_problem_statement)
        assert problem_statement == "Calculate the remainder when 93 is divided by 59."
        assert env.compute_graph.formal_elements == [Val("93"), Val("59")]
        # first action
        action = mod
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
            info["raw_observation"] == f"{problem_statement}; mod('p_0','p_1')"
        )
        assert reward == 0
        assert not done
        # next action
        action = "f0"
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
            info["raw_observation"]
            == f"{problem_statement}; mod(Val('93'),'p_1')"
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
            "max_sequence_length": 100,
            "vocab_size": 200
        }
        env = MathEnv(env_config)
        # reset - then succeed after 4th action
        encoded_problem_statement, _ = env.reset_with_specific_problem(
            "short_problems", 0, 6
        )
        problem_statement = env.decode(encoded_problem_statement)
        assert problem_statement == "Calculate the highest common divisor of 1300 and 300."
        # first action
        action = gcd
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
                info["raw_observation"] == f"{problem_statement}; gcd('p_0','p_1')"
        )
        assert reward == 0
        assert not done
        # next action
        action = "f0"
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
                info["raw_observation"]
                == f"{problem_statement}; gcd(Val('1300'),'p_1')"
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

    def test_problem_8_success_1(self):
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
            "short_problems", 0, 8
        )
        problem_statement = env.decode(encoded_problem_statement)
        assert problem_statement == "Is 93163 a prime number?"
        # first action
        action = ip
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
                info["raw_observation"] == f"{problem_statement}; ip('p_0')"
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

    def test_problem_8_success_2(self):
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
            "short_problems", 0, 9
        )
        problem_statement = env.decode(encoded_problem_statement)
        assert problem_statement == "Is 66574 a composite number?"
        # first action
        action = nt
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
                info["raw_observation"] == f"{problem_statement}; nt('p_0')"
        )
        assert reward == 0
        assert not done
        # next action
        action = ip
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
                info["raw_observation"]
                == f"{problem_statement}; nt(ip('p_0'))"
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

    def test_problem_9_success(self):
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
            "short_problems", 0, 9
        )
        problem_statement = env.decode(encoded_problem_statement)
        assert problem_statement == "Is 66574 a composite number?"
        # take action
        action = nt
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
                info["raw_observation"] == f"{problem_statement}; nt('p_0')"
        )
        assert reward == 0
        assert not done
        # take action
        action = nt
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
                info["raw_observation"] == f"{problem_statement}; nt(nt('p_0'))"
        )
        assert reward == 0
        assert not done

    def test_univariate(self):
        env_config = {
            "problem_filepaths": ["artifacts/calculus__differentiate.txt"],
            "corpus_filepath": "../../environment/corpus/1k_corpus.txt",
            "num_problems_per_module": 10 ** 7,
            "validation_percentage": 0,
            "max_sequence_length": 100,
            "vocab_size": 200,
            "univariate_differentiation": True
        }
        env = MathEnv(env_config)
        # for q in env.train['calculus__differentiate'][0][:20]:
        #     print(q)
