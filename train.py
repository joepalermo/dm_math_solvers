#!/usr/bin/env python3

import argparse
import functools
import itertools
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import typing
from copy import deepcopy
from functools import reduce
from logging import debug, info, log
from pathlib import Path

import gym
import numpy as np
import ray
import torch
import torch.nn.functional as F
from gym.spaces import Box, Discrete, MultiDiscrete
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.tf.attention_net import GTrXLNet
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune import grid_search
from torch import Tensor, distributions, nn, tensor
from torch.nn import Linear, ReLU, Sequential, Softmax
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, TensorDataset

from environment.envs.math_env import MathEnv

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

# TODO mathenv needs to take in dict `config` in __init__` (for ray to work

# "model": {
#     "custom_model": GTrXLNet,
#     "max_seq_len": 50,
#     "custom_model_config": {
#         "num_transformer_units": 1,
#         "attn_dim": 64,
#         "num_heads": 2,
#         "memory_tau": 50,
#         "head_dim": 32,
#         "ff_hidden_dim": 32,
#     },
# },

# TODO lightning module
class TorchCustomModel(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a GTrXLNet."""

    def __init__(
        self,
        obs_space: MultiDiscrete,
        action_space: Discrete,
        num_outputs: int,  # TODO should be action_space.n
        model_config: dict,
        name: str,
    ):
        TorchModelV2.__init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
        )
        nn.Module.__init__(self)

        self.torch_sub_model = GTrXLNet(
            obs_space, action_space, num_outputs, model_config, name,
            num_transformer_units=1,
            attn_dim=64,
            num_heads=2,
            memory_tau=50,
            head_dim=32,
            ff_hidden_dim=32)

    def forward(self, input_dict, state, seq_lens):
        input_dict["obs"] = input_dict["obs"].float()
        logits, _ = self.torch_sub_model(input_dict, state, seq_lens)
        return logits, []

    def value_function(self):
        return torch.reshape(self.torch_sub_model.value_function(), [-1])


if __name__ == "__main__":
    args = parser.parse_args()
    env_config = {
        "problem_filepaths": ['/Users/joe/workspace/projects/dm_math_solvers/mathematics_dataset-v1.0/train-easy/numbers__gcd.txt'],  # TODO hardcode single path to make this easy to run
        "corpus_filepath": "/Users/joe/workspace/projects/dm_math_solvers/environment/corpus/10k_corpus.txt",
        "num_problems_per_module": 10 ** 7,
        # data used for validation
        "p_val": 0.2,
    }

    ray.init()

    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))
    ModelCatalog.register_custom_model("my_model", TorchCustomModel)

    # config = {
    #     "env": MathEnv,  # or "corridor" if registered above
    #     "env_config": env_config,
    #     # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    #     "num_gpus": torch.cuda.device_count(),
    #     "model": {
    #         "custom_model": "my_model",
    #         "max_seq_len": 50  # TODO: should this be here?
    #     },
    #     "vf_share_layers": True,
    #     "lr": grid_search([1e-2]),  # try different lrs
    #     "num_workers": 1,  # parallelism
    #     "framework": "torch",
    # }

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
            "custom_model": GTrXLNet,
            "max_seq_len": 250,
            "custom_model_config": {
                "num_transformer_units": 1,
                "attn_dim": 64,
                "num_heads": 2,
                "memory_tau": 50,
                "head_dim": 32,
                "ff_hidden_dim": 32,
            },
        },
        "framework": "torch" if args.torch else "tf",
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
