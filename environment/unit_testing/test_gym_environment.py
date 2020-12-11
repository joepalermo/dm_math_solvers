import numpy as np
import gym
from environment.utils import extract_formal_elements
from environment.envs.math_env import MathEnv
from environment.typed_operators import lookup_value, solve_system, append, make_equality, lookup_value_eq, project_lhs, \
    substitution_left_to_right, extract_isolated_variable, factor, simplify, diff, replace_arg, make_function, append_to_empty_list
import unittest



class Test(unittest.TestCase):

    def test_problem_0_fail_1(self):
        env = MathEnv(['environment/unit_testing/artifacts/test_problems.txt'])
        # reset - then fail after 1st action
        observation = env.reset_by_index(0)
        f = extract_formal_elements(observation)  # for use below
        assert f == ['0 = 4*b + b + 15', 'b']
        assert observation == 'Solve 0 = 4*b + b + 15 for b.'
        action = 'f0'
        formal_element = env.compute_graph.lookup_formal_element(action)
        observation_, reward, done, _ = env.step(action)
        assert observation_ == f"{observation}; '{formal_element}'"
        assert reward == 0
        assert done

    def test_problem_0_fail_2(self):
        env = MathEnv(['environment/unit_testing/artifacts/test_problems.txt'])
        # reset - then fail after 2nd action
        observation = env.reset_by_index(0)
        assert observation == 'Solve 0 = 4*b + b + 15 for b.'
        action = solve_system
        observation_, reward, done, _ = env.step(action)
        assert observation_ == f"{observation}; solve_system('param_0')"
        assert reward == 0
        assert not done
        # assert that lookup_value & append are the only actions not masked
        policy_vector = env.sample_masked_policy_vector()
        np.testing.assert_equal(np.ceil(policy_vector), np.array([1,0,1,1,0,0]))
        # next action
        action = 'f0'
        observation_, reward, done, _ = env.step(action)
        assert observation_ == f"{observation}; solve_system('0 = 4*b + b + 15')"
        assert reward == 0
        assert done

    def test_problem_0_success_1(self):
        env = MathEnv(['environment/unit_testing/artifacts/test_problems.txt'])
        # reset - then succeed after 4th action
        observation = env.reset_by_index(0)
        assert observation == 'Solve 0 = 4*b + b + 15 for b.'
        action = lookup_value
        observation_, reward, done, _ = env.step(action)
        assert observation_ == f"{observation}; lookup_value('param_0','param_1')"
        assert reward == 0
        assert not done
        assert env.compute_graph.current_node == env.compute_graph.root
        # assert that lookup_value & solve_system are the only actions not masked
        # because dict: is object, is dict, is not list, is not Equation, is not Variable
        policy_vector = env.sample_masked_policy_vector()
        np.testing.assert_equal(np.ceil(policy_vector), np.array([1, 1, 0, 0, 0, 0]))
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

        #next action
        action = append_to_empty_list
        observation_, reward, done, _ = env.step(action)
        assert observation_ == f"{observation}; lookup_value(solve_system(append_to_empty_list('param_0')),'b')"
        assert reward == 0
        assert not done

        # next action
        action = 'f0'
        observation_, reward, done, _ = env.step(action)
        assert observation_ == f"{observation}; lookup_value(solve_system(append_to_empty_list('0 = 4*b + b + 15')),'b')"
        assert reward == 1
        assert done

    def test_guess_problem_0(self):
        '''this test only terminates when the graph is correctly guessed'''
        env = MathEnv(['environment/unit_testing/artifacts/test_problems.txt'])
        episode_i = 0
        graph_guessed_correctly = False
        while not graph_guessed_correctly:
            problem_statement = env.reset_by_index(0)
            observation = problem_statement
            done = False
            step_i = 0
            # print(f"episode: {episode_i}")
            while not done:
                action = env.sample_masked_action()
                # print(f"\tstep: {step_i}")
                # print(f"\t\tS: {observation}, A: {action}")
                observation, reward, done, _ = env.step(action)
                print(f"\t\tS': {observation}, R: {reward}, done: {done}")
                if reward == 1:
                    assert observation == f"{problem_statement}; lookup_value(solve_system(append_to_empty_list('0 = 4*b + b + 15')),'b')"
                    graph_guessed_correctly = True
                step_i += 1
            episode_i += 1
            print(episode_i)
        print(f'trials taken to guess problem 0: {episode_i}')

    def test_guess_problem_1(self):
        '''this test only terminates when the graph is correctly guessed'''
        env = MathEnv(['environment/unit_testing/artifacts/test_problems.txt'])
        episode_i = 0
        graph_guessed_correctly = False
        while not graph_guessed_correctly:
            problem_statement = env.reset_by_index(1)
            observation = problem_statement
            done = False
            step_i = 0
            # print(f"episode: {episode_i}")
            while not done:
                action = env.sample_masked_action()
                # print(f"\tstep: {step_i}")
                # print(f"\t\tS: {observation}, A: {action}")
                observation, reward, done, _ = env.step(action)
                print(f"\t\tS': {observation}, R: {reward}, done: {done}")
                if reward == 1:
                    #assert observation == f"{problem_statement}; lookup_value(solve_system(append_to_empty_list('0 = 4*b + b + 15')),'b')"
                    graph_guessed_correctly = True
                step_i += 1
            episode_i += 1
            print(episode_i)
        print(f'trials taken to guess problem 1: {episode_i}')

    def test_guess_problem_2(self):
        '''this test only terminates when the graph is correctly guessed'''
        env = MathEnv(['environment/unit_testing/artifacts/test_problems.txt'])
        episode_i = 0
        graph_guessed_correctly = False
        while not graph_guessed_correctly:
            problem_statement = env.reset_by_index(1)
            observation = problem_statement
            done = False
            step_i = 0
            # print(f"episode: {episode_i}")
            while not done:
                action = env.sample_masked_action()
                # print(f"\tstep: {step_i}")
                # print(f"\t\tS: {observation}, A: {action}")
                observation, reward, done, _ = env.step(action)
                print(f"\t\tS': {observation}, R: {reward}, done: {done}")
                if reward == 1:
                    #assert observation == f"{problem_statement}; lookup_value(solve_system(append_to_empty_list('0 = 4*b + b + 15')),'b')"
                    graph_guessed_correctly = True
                step_i += 1
            episode_i += 1
            print(episode_i)
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
