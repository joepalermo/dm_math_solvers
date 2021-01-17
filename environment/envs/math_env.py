from inspect import signature
from pathlib import Path
from random import sample

import gym
import numpy as np
from gym import spaces
from scipy.special import softmax
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from environment.compute_graph import ComputeGraph
from environment.typed_operators import *
from environment.utils import load_training_data, split_validation_data
from hparams import HParams

hparams = HParams.get_hparams_by_name('rl_math')

class MathEnv(gym.Env):
    def __init__(self, config):
        self.compute_graph = None
        # load config
        self.config = config
        self.max_num_nodes = config.max_num_nodes
        self.max_formal_elements = config.max_formal_elements
        self.max_difficulty = config.max_difficulty
        self.vocab_size = config.vocab_size
        # define available operator functions
        self.operators = [
            lv,
            ss,
            ap,
            ape,
            meq,
            lve,
            eiv,
            slr,
            fac,
            df,
            dfw,
            sy,
            mfn,
            ra,
            mod,
            gcd,
            md0,
            ip,
            lcm,
            lcd,
            pf,
            fa,
            nt
        ]
        # ensure that every operator listed in config.operators is present in the above list
        valid_op_names = [op.__name__ for op in self.operators]
        assert all([op in valid_op_names for op in config.operators])
        # define action and observation space
        self.operators = [operator for operator in self.operators if (operator.__name__ in config.operators)]
        self.operator_output_types = [
            signature(operator).return_annotation for operator in self.operators
        ]
        self.actions = self.operators + [
            f"f{i}" for i in range(self.max_formal_elements)
        ]
        self.action_space = spaces.Discrete(len(self.actions))
        self.action_indices = np.arange(len(self.actions))
        self.observation_space = spaces.MultiDiscrete(
            [self.vocab_size + 1 for _ in range(config.max_sequence_length)]  # increment by 1 for padding_token
        )
        # load data
        self.train = load_training_data(config)
        self.val = split_validation_data(config, self.train)
        # build tokenizer
        self.padding_token = self.vocab_size
        self.special_tokens = [operator.__name__ for operator in self.operators] + ["'p_0'", "'p_1'"]
        self.tokenizer = self.build_tokenizer()

    def step(self, action_index):
        """
        :param action_index: index into the action space
        :return: observation, reward, done, info

        An action fills the next element in the compute graph.
        -observation: problem statement + interim compute graph
        -reward: 0 if the compute doesn't evaluate correctly, 1 if it does
        -done: True if the graph is complete, False if it isn't
        -info: None
        """
        action = self.actions[action_index]
        self.compute_graph.n_nodes += 1
        self.compute_graph.add(action)
        output = self.compute_graph.eval()
        compute_graph = str(self.compute_graph)
        raw_observation = f"{self.problem_statement}; {compute_graph}"
        observation = self.encode(raw_observation)
        done = (
            self.compute_graph.current_node is None
            or self.compute_graph.n_nodes >= self.max_num_nodes
        )
        # get reward
        if done and str(output) == self.answer:
            reward = 1
        elif done:
            reward = -1
        else:
            reward = 0
        info = {"raw_observation": raw_observation}
        return observation, reward, done, info

    # tokenization utilities -------------------------------------------------------------------------------------------

    def build_tokenizer(self):
        tokenizer = Tokenizer(BPE())
        trainer = BpeTrainer(vocab_size=self.vocab_size, special_tokens=self.special_tokens)
        print(self.config.corpus_filepath)
        tokenizer.train(trainer, [str(Path(self.config.corpus_filepath).resolve())])
        return tokenizer

    def encode(self, raw_observation):
        encoded_ids = self.tokenizer.encode(raw_observation).ids
        # pad the encoded ids up to a maximum length
        encoded_ids.extend(
            [self.padding_token for _ in range(self.config.max_sequence_length - len(encoded_ids))]
        )
        return np.array(encoded_ids)

    def decode(self, ids):
        # filter out padding tokens before decoding
        ids = [id_ for id_ in ids if id_ != self.padding_token]
        return "".join([self.tokenizer.id_to_token(id_) for id_ in ids]).strip()

    # utilities to reset the environment -------------------------------------------------------------------------------

    def reset(self, train=True):
        # randomly sample a module and difficulty level
        self.module_name = sample(list(self.train.keys()), 1)[0]
        self.difficulty = sample(list(self.train[self.module_name].keys()), 1)[0]
        return self.reset_by_module_and_difficulty(self.module_name, self.difficulty, train=train)

    def reset_with_same_problem(self):
        self.compute_graph = ComputeGraph(self.problem_statement)
        return self.encode(self.problem_statement), {'raw_observation': self.problem_statement}

    def reset_with_specific_problem(
        self, module_name, difficulty, problem_index, train=True
    ):
        self.module_name = module_name
        self.difficulty = difficulty
        if train:
            self.problem_statement, self.answer = self.train[module_name][difficulty][
                problem_index
            ]
        else:
            self.problem_statement, self.answer = self.val[module_name][difficulty][
                problem_index
            ]
        self.compute_graph = ComputeGraph(self.problem_statement)
        return self.encode(self.problem_statement), {'raw_observation': self.problem_statement}

    def reset_by_module_and_difficulty(self, module_name, difficulty, train=True):
        self.module_name = module_name
        self.difficulty = difficulty
        if train:
            self.problem_statement, self.answer = sample(
                self.train[module_name][difficulty], 1
            )[0]
        else:
            self.problem_statement, self.answer = sample(
                self.val[module_name][difficulty], 1
            )[0]
        self.compute_graph = ComputeGraph(self.problem_statement)
        return self.encode(self.problem_statement), {'raw_observation': self.problem_statement}

    # utilities to sample actions --------------------------------------------------------------------------------------

    def get_action_index(self, action):
        return self.actions.index(action)

    def sample_action_index(self):
        return self.action_space.sample()

    def sample_masked_action_index(self):
        choices = np.arange(len(self.actions))
        masked_policy_vector = self.sample_masked_policy_vector()
        return np.random.choice(choices, p=masked_policy_vector)

    def sample_masked_policy_vector(self):
        policy_vector = np.random.uniform(size=len(self.actions))
        masked_policy_vector = self.mask_invalid_types(policy_vector)
        masked_normed_policy_vector = masked_policy_vector / np.sum(
            masked_policy_vector
        )
        return masked_normed_policy_vector

    def sample_masked_action_from_model(self, model, obs):
        policy_vector = softmax(model(obs).detach().numpy()[0])
        masked_policy_vector = self.mask_invalid_types(policy_vector)
        masked_normed_policy_vector = masked_policy_vector / np.sum(
            masked_policy_vector
        )
        choices = np.arange(len(self.actions))
        action_index = np.random.choice(choices, p=masked_normed_policy_vector)
        return action_index

    def mask_invalid_types(self, policy_vector):
        if not self.compute_graph.current_node:
            # first action must be an operator
            mask = np.concatenate(
                [np.ones(len(self.operators)), np.zeros(self.max_formal_elements)]
            )
        else:
            current_arg_index = len(self.compute_graph.current_node.args)
            next_type = self.compute_graph.current_node.types[current_arg_index]
            available_types = (
                self.operator_output_types + self.compute_graph.formal_element_types
            )
            mask = np.array(
                [1 if issubclass(type_, next_type) else 0 for type_ in available_types]
            )
            mask = np.concatenate(
                [
                    mask,
                    np.zeros(
                        self.max_formal_elements
                        - len(self.compute_graph.formal_elements)
                    ),
                ]
            )
        return mask * policy_vector

    def render(self):
        pass

    def close(self):
        pass
