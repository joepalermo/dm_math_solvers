import gym
from environment.utils import extract_formal_elements
from environment.envs.math_env import MathEnv
from environment.operators import append, add_keypair, lookup_value, function_application, apply_mapping, calc, \
    make_equality, project_lhs, project_rhs, simplify, solve_system, factor, diff, replace_arg, substitution_left_to_right, \
    eval_in_base, root, round_to_int, round_to_dec, power, substitution_right_to_left, max_arg, min_arg, greater_than, \
    less_than, lookup_value_eq
import unittest

class Test(unittest.TestCase):

    def test_problem_0(self):
        env = MathEnv(['environment/unit_testing/test_problems.txt'])
        # reset - then fail after 1st action
        observation = env.reset_by_index(0)
        f = extract_formal_elements(observation)  # for use below
        assert observation == 'Solve $f[0 = 4*b + b + 15] for $f[b].'
        action = 'f0'
        formal_element = env.compute_graph.lookup_formal_element(action)
        observation_, reward, done, _ = env.step(action)
        assert observation_ == f"{observation}; '{formal_element}'"
        assert reward == 0
        assert done
        # reset - then fail after 2nd action
        observation = env.reset_by_index(0)
        assert observation == 'Solve $f[0 = 4*b + b + 15] for $f[b].'
        action = solve_system
        observation_, reward, done, _ = env.step(action)
        assert observation_ == f"{observation}; solve_system('param_0')"
        assert reward == 0
        assert not done
        # next action
        action = 'f0'
        observation_, reward, done, _ = env.step(action)
        assert observation_ == f"{observation}; solve_system('0 = 4*b + b + 15')"
        assert reward == 0
        assert done
        # reset - then succeed after 4th action
        observation = env.reset_by_index(0)
        assert observation == 'Solve $f[0 = 4*b + b + 15] for $f[b].'
        action = lookup_value
        observation_, reward, done, _ = env.step(action)
        assert observation_ == f"{observation}; lookup_value('param_0','param_1')"
        assert reward == 0
        assert not done
        assert env.compute_graph.current_node == env.compute_graph.root
        # next action
        action = solve_system
        observation_, reward, done, _ = env.step(action)
        assert observation_ == f"{observation}; lookup_value(solve_system('param_0'),'param_1')"
        assert reward == 0
        assert not done
        # current node is still root because it takes 2 arguments and only 1 has been given
        assert env.compute_graph.current_node == env.compute_graph.root
        # next action
        action = 'f1'
        observation_, reward, done, _ = env.step(action)
        assert observation_ == f"{observation}; lookup_value(solve_system('param_0'),'b')"
        assert reward == 0
        assert not done
        # current node is now the solve_system node because the lookup_value node has its args set
        assert env.compute_graph.current_node == env.compute_graph.root.args[0]
        # next action
        action = 'f0'
        observation_, reward, done, _ = env.step(action)
        assert observation_ == f"{observation}; lookup_value(solve_system('0 = 4*b + b + 15'),'b')"
        assert reward == 1
        assert done

    def test_problem_1(self):
        env = MathEnv(['environment/unit_testing/test_problems.txt'])
        # reset - then succeed after 2nd action
        observation = env.reset_by_index(1)  # select short dummy problem
        f = extract_formal_elements(observation)  # for use below
        assert observation == 'Solve $f[0 = 4*b + b + 15] for $f[b].'
        action = solve_system
        observation_, reward, done, _ = env.step(action)
        assert observation_ == f"{observation}; solve_system('param_0')"
        assert reward == 0
        assert not done
        # next action
        action = 'f0'
        observation_, reward, done, _ = env.step(action)
        assert observation_ == f"{observation}; solve_system('0 = 4*b + b + 15')"
        assert reward == 1
        assert done

    def test_guess_problem_0(self):
        '''this test only terminates when the graph is correctly guessed'''
        env = MathEnv(['environment/unit_testing/test_problems.txt'])
        episode_i = 0
        graph_guessed_correctly = False
        while not graph_guessed_correctly:
            problem_statement = env.reset_by_index(0)
            observation = problem_statement
            done = False
            step_i = 0
            # print(f"episode: {episode_i}")
            while not done:
                action = env.sample_action()
                # print(f"\tstep: {step_i}")
                # print(f"\t\tS: {observation}, A: {action}")
                observation, reward, done, _ = env.step(action)
                # print(f"\t\tS': {observation}, R: {reward}, done: {done}")
                if reward == 1:
                    assert observation == f"{problem_statement}; lookup_value(solve_system('0 = 4*b + b + 15'),'b')"
                    graph_guessed_correctly = True
                step_i += 1
            episode_i += 1
        print(f'trials taken to guess problem 0: {episode_i}')

    def test_guess_problem_1(self):
        '''this test only terminates when the graph is correctly guessed'''
        env = MathEnv(['environment/unit_testing/test_problems.txt'])
        episode_i = 0
        graph_guessed_correctly = False
        while not graph_guessed_correctly:
            problem_statement = env.reset_by_index(1)
            observation = problem_statement
            done = False
            step_i = 0
            # print(f"episode: {episode_i}")
            while not done:
                action = env.sample_action()
                # print(f"\tstep: {step_i}")
                # print(f"\t\tS: {observation}, A: {action}")
                observation, reward, done, _ = env.step(action)
                # print(f"\t\tS': {observation}, R: {reward}, done: {done}")
                if reward == 1:
                    assert observation == f"{problem_statement}; solve_system('0 = 4*b + b + 15')"
                    graph_guessed_correctly = True
                step_i += 1
            episode_i += 1
        print(f'trials taken to guess problem 1: {episode_i}')


    # def test(self):
    #     # env = gym.make('math')
    #     env = MathEnv(['mathematics_dataset-v1.0/train-easy/algebra__linear_1d.txt'])
    #     for _ in range(3):
    #         # env.render()
    #         env.step(env.action_space.sample())  # take a random action
    #         observation = env.reset()
    #         print(f'observation: {observation}')
    #         action = env.actions[env.action_space.sample()]
    #         # action = 'f0'
    #         # action = solve_system
    #         print(f'action: {action}')
    #         observation, reward, done, info = env.step(action)
    #         print(f'observation: {observation}')
    #         print(f'reward: {reward}')
    #         print(f'done: {done}')
    #         # print(observation, reward, done, info)
    #     env.close()
