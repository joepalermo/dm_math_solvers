from environment.envs.math_env import MathEnv

math_env = MathEnv(['mathematics_dataset-v1.0/train-easy/algebra__linear_1d.txt'])
print(math_env.problems[:10])