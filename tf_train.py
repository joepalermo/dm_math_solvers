#!/usr/bin/env python3

import argparse
import os
import ray
from ray import tune
from ray.rllib.utils.test_utils import check_learning_achieved
from environment.envs.math_env import MathEnv
from tf_transformer_encoder import Encoder
from transformer.transformer_utils import create_padding_mask
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
import tensorflow as tf
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
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--stop-iters", type=int, default=50)
parser.add_argument("--stop-timesteps", type=int, default=100000)
parser.add_argument("--stop-reward", type=float, default=0.1)


class TransformerEncoder(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, num_layers, d_model, num_heads, dff, vocab_size, seq_len, attention_dropout):
        super(TransformerEncoder, self).__init__(obs_space, action_space,
                                           num_outputs, model_config, name)
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, vocab_size, seq_len, attention_dropout)
        self.policy_layer = tf.keras.layers.Dense(3)
        self.value_layer = tf.keras.layers.Dense(1)

    def call(self, inp, enc_padding_mask):
        enc_output = self.encoder(inp, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
        select_first_pos = tf.keras.layers.Lambda(lambda x: x[:, 0, :])
        policy_output = self.policy_layer(select_first_pos(enc_output))  # (batch_size, 3)
        value_output = self.value_layer(select_first_pos(enc_output))  # (batch_size, 1)
        return policy_output, value_output

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        enc_padding_mask = create_padding_mask(obs)
        model_out, self._value_out = self.call(obs, enc_padding_mask)
        return model_out, state

    def value_function(self):
        return self._value_out
        # return tf.reshape(self._value_out, [-1])

    def metrics(self):
        return {"foo": tf.constant(42.0)}


if __name__ == "__main__":
    args = parser.parse_args()
    env_config = {
        "problem_filepaths": [Path('mathematics_dataset-v1.0/train-easy/numbers__gcd.txt')],  # TODO hardcode single path to make this easy to run
        "corpus_filepath": Path("environment/corpus/10k_corpus.txt"),
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
            "custom_model": TransformerEncoder,
            "custom_model_config": {
                "num_layers": 1,
                "d_model": 256,
                "num_heads": 4,
                "dff": 256,
                "vocab_size": 280,
                "seq_len": 250,
                "attention_dropout": 0.1
            },
        },
        "framework": "tf",
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
