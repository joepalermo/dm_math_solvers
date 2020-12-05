import gym
from gym import error, spaces, utils
from gym.utils import seeding
from environment.operators import append, add_keypair, lookup_value, function_application, apply_mapping, calc, \
    make_equality, project_lhs, project_rhs, simplify, solve_system, factor, diff, replace_arg, substitution_left_to_right, \
    eval_in_base, root, round_to_int, round_to_dec, power, substitution_right_to_left, max_arg, min_arg, greater_than, \
    less_than, lookup_value_eq
from environment.compute_graph import ComputeGraph


class MathEnv(gym.Env):

    def __init__(self):
        self.operators = [append, lookup_value, solve_system]
        self.max_formal_elements = 5
        self.actions = self.operators + [f"f{i}" for i in range(self.max_formal_elements)]

        self.problems = []
        self.problem = None
        self.compute_graph = ComputeGraph()

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
        self.compute_graph.add(action)
        output = self.compute_graph.eval()
        observation = self.problem + str(self.compute_graph)
        reward = 1 if output == self.answer else 0
        done = output is not None
        info = {}
        return observation, reward, done, info

    def reset(self):
        '''
        resets the environment by sampling a new problem.
        :return: the initial oberservation (the problem statement)
        '''
        self.compute_graph.reset()
        self.problem_statement, self.answer = self.problems.sample()
        return self.problem

    def render(self):
        pass

    def close(self):
        pass