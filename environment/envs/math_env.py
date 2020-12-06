import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym import spaces
from environment.operators import append, add_keypair, lookup_value, function_application, apply_mapping, calc, \
    make_equality, project_lhs, project_rhs, simplify, solve_system, factor, diff, replace_arg, substitution_left_to_right, \
    eval_in_base, root, round_to_int, round_to_dec, power, substitution_right_to_left, max_arg, min_arg, greater_than, \
    less_than, lookup_value_eq
from environment.compute_graph import ComputeGraph
from random import sample


class MathEnv(gym.Env):

    def __init__(self, problem_filepaths):
        self.operators = [lookup_value, solve_system]
        self.max_formal_elements = 2
        self.actions = self.operators + [f"f{i}" for i in range(self.max_formal_elements)]
        self.action_space = spaces.Discrete(len(self.actions))
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
        compute_graph = str(self.compute_graph)
        observation = f"{self.problem_statement}; {compute_graph}"
        reward = 1 if str(output) == self.answer else 0
        done = True if not self.compute_graph.current_node else False
        info = {}
        return observation, reward, done, info

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