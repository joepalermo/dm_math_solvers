#!/usr/bin/env python3

import argparse
import os
import ray
from ray import tune
from ray.rllib.utils.test_utils import check_learning_achieved
from environment.envs.math_env import MathEnv
from transformer_encoder import TransformerModel
from pathlib import Path

"""Example of a custom gym environment and model. Run this for a demo.

This example shows:
  - using a custom environment
  - using a custom model
  - using Tune for grid search

You can visualize experiment results in ~/ray_results using TensorBoard.
"""

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")
parser.add_argument("--torch", action="store_true")
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--stop-iters", type=int, default=50)
parser.add_argument("--stop-timesteps", type=int, default=100000)
parser.add_argument("--stop-reward", type=float, default=0.1)

if __name__ == "__main__":
    args = parser.parse_args()
    env_config = {
        "problem_filepaths": [
            str(Path("mathematics_dataset-v1.0/train-easy/numbers__gcd.txt").resolve())
        ],  # TODO hardcode single path to make this easy to run
        "corpus_filepath": str(Path("environment/corpus/10k_corpus.txt").resolve()),
        "num_problems_per_module": 10 ** 7,
        # data used for validation
        "p_val": 0.2,
    }

    ray.init()

    config = {
        "env": MathEnv,
        "env_config": env_config,
        "gamma": 0.99,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", 0)),
        "num_workers": 0,
        "num_envs_per_worker": 1,
        "entropy_coeff": 0.001,
        "num_sgd_iter": 5,
        "vf_loss_coeff": 1e-5,
        "model": {
            "custom_model": TransformerModel,
            "custom_model_config": {
                "ntoken": 280,
                "ninp": 250,
                "nhead": 4,
                "nhid": 256,
                "nlayers": 1,
            },
        },
        "framework": "torch",
    }

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    results = tune.run(args.run, config=config, stop=stop)

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)
    ray.shutdown()
