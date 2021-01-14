from inspect import signature
from random import sample

import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from tqdm import tqdm
from scipy.special import softmax
from environment.compute_graph import ComputeGraph
from environment.typed_operators import *
from environment.utils import filter_univariate
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from utils import write_pickle, read_pickle
from hparams import HParams
hparams = HParams.get_hparams_by_name('rl_math')

class MathEnv(gym.Env):
    def __init__(self, config):
        self.config = config
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
        self.operators = [operator for operator in self.operators if (operator.__name__ in hparams.env.operators)]
        self.operator_output_types = [
            signature(operator).return_annotation for operator in self.operators
        ]
        self.max_formal_elements = 13  # TODO: make into a hyperparameter
        self.actions = self.operators + [
            f"f{i}" for i in range(self.max_formal_elements)
        ]
        self.max_n_nodes = 10  # TODO: make into a hyperparameter
        self.action_space = spaces.Discrete(len(self.actions))
        self.action_indices = np.arange(len(self.actions))
        self.vocab_size = config["vocab_size"]
        self.observation_space = spaces.MultiDiscrete(
            [self.vocab_size + 1 for _ in range(self.config["max_sequence_length"])]  # increment by 1 for padding_token
        )
        # load train data
        self.max_difficulty = self.config.get("max_difficulty", 99)  # by default set max_difficulty very high
        self.train = {}
        print("loading problems")
        for filepath in tqdm(self.config["problem_filepaths"]):
            module_name = filepath.split("/")[-1].split(".txt")[0]
            if "compose" in module_name:
                compose = True
                module_name = module_name.split("_compose")[0]
            else:
                compose = False
                module_name = module_name
            with open(filepath, "r") as f:
                lines = f.readlines()
            num_pairs = min(len(lines) // 2, self.config["num_problems_per_module"])
            for i in range(0, 2 * num_pairs, 2):
                question = lines[i].strip()
                answer = lines[i + 1].strip()
                # for uncomposed problems set difficulty to 0 to distinguish them
                difficulty = (
                    len(re.split("(?<![0-9])[.,;:?]|[.,;:?](?![0-9])", question)) - 1
                    if compose
                    else 0
                )
                # don't load problems with difficulty above the maximum
                if difficulty > self.max_difficulty:
                    continue
                if module_name in self.train:
                    if difficulty in self.train[module_name]:
                        self.train[module_name][difficulty].append((question, answer))
                    else:
                        self.train[module_name][difficulty] = [(question, answer)]
                else:
                    self.train[module_name] = {difficulty: [(question, answer)]}
        if config["univariate_differentiation"]:
            self.train['calculus__differentiate'][0] = filter_univariate(self.train['calculus__differentiate'][0])
        # split out val data
        self.val = {}
        for module_name in self.train:
            self.val[module_name] = {}
            for difficulty in self.train[module_name]:
                num_examples = len(self.train[module_name][difficulty])
                num_val = int(num_examples * self.config["validation_percentage"])
                self.val[module_name][difficulty] = self.train[module_name][difficulty][
                    :num_val
                ]
                self.train[module_name][difficulty] = self.train[module_name][
                    difficulty
                ][num_val:]
                assert (
                    len(self.train[module_name][difficulty])
                    + len(self.val[module_name][difficulty])
                    == num_examples
                )
        # TODO: load test data
        self.compute_graph = None
        # build or load encoder
        self.padding_token = self.vocab_size
        self.special_tokens = [operator.__name__ for operator in self.operators] + ["'p_0'", "'p_1'"]
        self.tokenizer = Tokenizer(BPE())
        trainer = BpeTrainer(vocab_size=self.vocab_size, special_tokens=self.special_tokens)
        self.tokenizer.train(trainer, [self.config["corpus_filepath"]])

    def encode(self, raw_observation):
        encoded_ids = self.tokenizer.encode(raw_observation).ids
        # pad the encoded ids up to a maximum length
        encoded_ids.extend(
            [self.padding_token for _ in range(self.config["max_sequence_length"] - len(encoded_ids))]
        )
        return np.array(encoded_ids)

    def decode(self, ids):
        # filter out padding tokens before decoding
        ids = [id_ for id_ in ids if id_ != self.padding_token]
        return "".join([self.tokenizer.id_to_token(id_) for id_ in ids]).strip()

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

    def step(self, action_index):
        """an action fills the next element in the compute graph.

        :param action_index: index into the action space
        :return: observation, reward, done, info

        -observation: problem statement + interim compute graph
        -reward: 0 if the compute doesn't evaluate correctly, 1 if it does
        -done: True if the graph is complete, False if it isn't
        -info: None
        """
        # (alok): TODO obs space should be multidiscrete?
        # (alok): TODO discrete (list of operators we're using)
        action = self.actions[action_index]
        self.compute_graph.n_nodes += 1
        self.compute_graph.add(action)
        output = self.compute_graph.eval()
        compute_graph = str(self.compute_graph)
        raw_observation = f"{self.problem_statement}; {compute_graph}"
        observation = self.encode(raw_observation)
        done = (
            self.compute_graph.current_node is None
            or self.compute_graph.n_nodes >= self.max_n_nodes
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
