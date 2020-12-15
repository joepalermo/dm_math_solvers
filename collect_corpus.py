from utils import read_text_file
from environment.envs.math_env import MathEnv


def guess_until_problem_solved(env, problem_index, verbose=False, max_episode_index=1000):
    episode_i = 0
    graph_guessed_correctly = False
    print(f'problem statement: {env.reset_with_specific_problem("short_problems", 1, problem_index)}')
    while not graph_guessed_correctly and episode_i < max_episode_index:
        _ = env.reset_with_specific_problem('short_problems', 1, problem_index)
        done = False
        step_i = 0
        if verbose:
            print(f"episode: {episode_i}")
        while not done:
            action = env.sample_masked_action()
            observation, reward, done, _ = env.step(action)
            # if verbose:
            #     print(f"\t\tS': {observation}, R: {reward}, done: {done}")
            if reward == 1:
                graph_guessed_correctly = True
            step_i += 1
        episode_i += 1
    print(f'graph: {observation.split(";")[1]}')
    print(f'trials taken to guess problem #{problem_index}: {episode_i}')


filenames = read_text_file('environment/module_lists/composed.txt').split('\n')
filepaths = [f'mathematics_dataset-v1.0/train-easy/{filename}' for filename in filenames]
env = MathEnv(filepaths, num_problems_per_module=int(1e4))
for i in range(1000):
    problem_statement = env.reset()
    print(problem_statement)
