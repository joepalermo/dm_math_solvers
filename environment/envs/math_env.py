import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym import spaces
from environment.typed_operators import lookup_value, solve_system, append, make_equality, lookup_value_eq, project_lhs, \
    substitution_left_to_right, substitution_right_to_left, extract_isolated_variable, factor, simplify, diff, replace_arg, \
    make_function, append_to_empty_list, mod, gcd, mod_eq_0, is_prime, lcm, prime_factors
from environment.compute_graph import ComputeGraph
from random import sample
from inspect import signature


class MathEnv(gym.Env):

    def __init__(self, problem_filepaths):
        self.operators = [lookup_value, solve_system, append, append_to_empty_list, make_equality, lookup_value_eq,
                          extract_isolated_variable, substitution_left_to_right, factor, diff, simplify, make_function,
                          replace_arg, mod, gcd, mod_eq_0, is_prime, lcm, prime_factors]  # TODO: make into a hyperparameter
        self.operator_output_types = [signature(operator).return_annotation for operator in self.operators]
        self.max_formal_elements = 6  # TODO: make into a hyperparameter
        self.actions = self.operators + [f"f{i}" for i in range(self.max_formal_elements)]
        self.action_space = spaces.Discrete(len(self.actions))
        self.max_n_nodes = 20
        # load problems
        self.problems = []
        for filepath in problem_filepaths:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            num_pairs = len(lines) // 2
            for i in range(0, 2 * num_pairs, 2):
                question = lines[i].strip()
                answer = lines[i + 1].strip()
                self.problems.append((question, answer))
        self.compute_graph = None

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
        masked_normed_policy_vector = masked_policy_vector / np.sum(masked_policy_vector)
        return masked_normed_policy_vector

    def step(self, action):
        '''
        an action fills the next element in the compute graph.

        :param action: an operator or a formal element
        :return: observation, reward, done, info

        -observation: problem statement + interim compute graph
        -reward: 0 if the compute doesn't evaluate correctly, 1 if it does
        -done: True if the graph is complete, False if it isn't
        -info: None
        '''
        self.compute_graph.n_nodes += 1
        self.compute_graph.add(action)
        output = self.compute_graph.eval()
        compute_graph = str(self.compute_graph)
        observation = f"{self.problem_statement}; {compute_graph}"
        reward = 1 if str(output) == self.answer else 0
        done = self.compute_graph.current_node is None or self.compute_graph.n_nodes > self.max_n_nodes
        info = {}
        return observation, reward, done, info

    def mask_invalid_types(self, policy_vector):
        if not self.compute_graph.current_node:
            mask = np.concatenate([np.ones(len(self.operators)), np.zeros(self.max_formal_elements)])
        else:
            current_arg_index = len(self.compute_graph.current_node.args)
            next_type = self.compute_graph.current_node.types[current_arg_index]
            available_types = self.operator_output_types + self.compute_graph.formal_element_types
            mask = np.array([1 if issubclass(next_type, type_) else 0 for type_ in available_types])
            mask = np.concatenate([mask, np.zeros(self.max_formal_elements - len(self.compute_graph.formal_elements))])
        return mask * policy_vector

    def reset(self):
        '''
        resets the environment by sampling a new problem.
        :return: the initial oberservation (the problem statement)
        '''
        self.problem_statement, self.answer = sample(self.problems, 1)[0]
        self.compute_graph = ComputeGraph(self.problem_statement)
        return self.problem_statement

    def reset_by_index(self, index):
        self.problem_statement, self.answer = self.problems[index]
        self.compute_graph = ComputeGraph(self.problem_statement)
        return self.problem_statement

    def render(self):
        pass

    def close(self):
        pass