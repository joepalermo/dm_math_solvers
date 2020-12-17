from inspect import signature
from random import sample

import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from tqdm import tqdm

from environment.compute_graph import ComputeGraph
from environment.typed_operators import *
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from utils import write_pickle, read_pickle


def decode(tokenizer, ids):
    return "".join([tokenizer.id_to_token(id_) for id_ in ids])



class MathEnv(gym.Env):
    def __init__(self, config):
        self.config = config
        self.operators = [
            lookup_value,
            solve_system,
            append,
            append_to_empty_list,
            make_equality,
            lookup_value_eq,
            extract_isolated_variable,
            substitution_left_to_right,
            factor,
            diff,
            simplify,
            make_function,
            replace_arg,
            mod,
            gcd,
            mod_eq_0,
            is_prime,
            lcm,
            prime_factors,
            function_application,
        ]  # TODO: make into a hyperparameter
        self.operator_output_types = [
            signature(operator).return_annotation for operator in self.operators
        ]
        self.max_formal_elements = 13  # TODO: make into a hyperparameter
        self.actions = self.operators + [
            f"f{i}" for i in range(self.max_formal_elements)
        ]
        self.action_space = spaces.Discrete(len(self.actions))
        self.max_n_nodes = 15
        # load train data
        self.train = {}
        print("loading problems")
        for filepath in tqdm(self.config["problem_filepaths"]):
            module_name = filepath.split("/")[-1].split(".txt")[0]
            if "compose" in module_name:
                compose = True
                module_type = module_name.split("_compose")[0]
            else:
                compose = False
                module_type = module_name
            with open(filepath, "r") as f:
                lines = f.readlines()
            num_pairs = min(len(lines) // 2, self.config['num_problems_per_module'])
            for i in range(0, 2 * num_pairs, 2):
                question = lines[i].strip()
                answer = lines[i + 1].strip()
                difficulty = (
                    len(re.split("(?<![0-9])[.,;:?]|[.,;:?](?![0-9])", question)) - 1
                    if compose
                    else 1
                )
                if module_type in self.train:
                    if difficulty in self.train[module_type]:
                        self.train[module_type][difficulty].append((question, answer))
                    else:
                        self.train[module_type][difficulty] = [(question, answer)]
                else:
                    self.train[module_type] = {difficulty: [(question, answer)]}
        # split out val data
        self.val = {}
        for module_type in self.train:
            self.val[module_type] = {}
            for difficulty in self.train[module_type]:
                num_examples = len(self.train[module_type][difficulty])
                num_val = int(num_examples * self.config['p_val'])
                self.val[module_type][difficulty] = self.train[module_type][difficulty][
                    :num_val
                ]
                self.train[module_type][difficulty] = self.train[module_type][
                    difficulty
                ][num_val:]
                assert (
                    len(self.train[module_type][difficulty])
                    + len(self.val[module_type][difficulty])
                    == num_examples
                )
        # TODO: load test data
        self.compute_graph = None
        # build or load encoder
        self.tokenizer = Tokenizer(BPE())
        trainer = BpeTrainer(vocab_size=280)
        self.tokenizer.train(trainer, [self.config['corpus_filepath']])

    def sample_action(self):
        return self.actions[self.action_space.sample()]

    def sample_masked_action(self):
        choices = np.arange(len(self.actions))
        masked_policy_vector = self.sample_masked_policy_vector()
        choice = np.random.choice(choices, p=masked_policy_vector)
        return self.actions[choice]

    def sample_masked_policy_vector(self):
        policy_vector = np.random.uniform(size=len(self.actions))
        masked_policy_vector = self.mask_invalid_types(policy_vector)
        masked_normed_policy_vector = masked_policy_vector / np.sum(
            masked_policy_vector
        )
        return masked_normed_policy_vector

    def step(self, action):
        """an action fills the next element in the compute graph.

        :param action: an operator or a formal element
        :return: observation, reward, done, info

        -observation: problem statement + interim compute graph
        -reward: 0 if the compute doesn't evaluate correctly, 1 if it does
        -done: True if the graph is complete, False if it isn't
        -info: None
        """
        #(alok): TODO obs space should be multidiscrete?
        #(alok): TODO discrete (list of operators we're using)
        self.compute_graph.n_nodes += 1
        self.compute_graph.add(action)
        output = self.compute_graph.eval()
        compute_graph = str(self.compute_graph)
        raw_observation = f"{self.problem_statement}; {compute_graph}"
        observation = self.tokenizer.encode(raw_observation).ids
        reward = 1 if str(output) == self.answer else 0
        done = (
            self.compute_graph.current_node is None
            or self.compute_graph.n_nodes > self.max_n_nodes
        )
        info = {'raw_observation': raw_observation}
        return observation, reward, done, info

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
                [1 if issubclass(next_type, type_) else 0 for type_ in available_types]
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

    def reset(self, train=True):
        # randomly sample a module and difficulty level
        module_type = sample(list(self.train.keys()), 1)[0]
        difficulty = sample(list(self.train[module_type].keys()), 1)[0]
        return self.reset_by_module_and_difficulty(module_type, difficulty)

    def reset_with_specific_problem(
        self, module_type, difficulty, problem_index, train=True
    ):
        self.problem_statement, self.answer = self.train[module_type][difficulty][
            problem_index
        ]
        self.compute_graph = ComputeGraph(self.problem_statement)
        return self.problem_statement

    def reset_by_module_and_difficulty(self, module_type, difficulty, train=True):
        self.problem_statement, self.answer = sample(
            self.train[module_type][difficulty], 1
        )[0]
        self.compute_graph = ComputeGraph(self.problem_statement)
        return self.problem_statement

    def render(self):
        pass

    def close(self):
        pass
