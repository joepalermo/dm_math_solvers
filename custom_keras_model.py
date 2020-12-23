"""Example of using a custom ModelV2 Keras-style model."""

import argparse
import os
import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf
from environment.envs.math_env import MathEnv
from tf_transformer_encoder import Encoder
from pathlib import Path


tf1, tf, tfv = try_import_tf()
parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PG")  # Try PG, PPO, DQN
parser.add_argument("--stop", type=int, default=200)
parser.add_argument("--use-vision-network", action="store_true")
parser.add_argument("--num-cpus", type=int, default=0)


class TransformerEncoder(tf.keras.Model):
    def __init__(self, params):
        super(TransformerEncoder, self).__init__()

        # Model config
        self.params = params
        num_layers = params.num_layers
        d_model = params.d_model
        num_heads = params.num_heads
        dff = params.dff
        vocab_size = params.vocab_size
        pe_input = params.questions_max_length
        pe_target = params.answer_max_length
        attention_dropout = params.attention_dropout

        # Instantiate the model
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, vocab_size, pe_input, attention_dropout)
        self.policy_layer = tf.keras.layers.Dense(3)
        self.value_layer = tf.keras.layers.Dense(1)

    def call(self, inp, enc_padding_mask):
        enc_output = self.encoder(inp, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
        select_first_pos = tf.keras.layers.Lambda(lambda x: x[:, 0, :])
        policy_output = self.policy_layer(select_first_pos(enc_output))  # (batch_size, 3)
        value_output = self.value_layer(select_first_pos(enc_output))  # (batch_size, 1)
        return policy_output, value_output


class MyKerasModel(TFModelV2):
    """Custom model for policy gradient algorithms."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(MyKerasModel, self).__init__(obs_space, action_space,
                                           num_outputs, model_config, name)
        self.inputs = tf.keras.layers.Input(
            shape=obs_space.shape, name="observations")
        layer_1 = tf.keras.layers.Dense(
            128,
            name="my_layer1",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0))(self.inputs)
        layer_out = tf.keras.layers.Dense(
            num_outputs,
            name="my_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(layer_1)
        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(layer_1)
        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def metrics(self):
        return {"foo": tf.constant(42.0)}


if __name__ == "__main__":
    env_config = {
        "problem_filepaths": [Path('mathematics_dataset-v1.0/train-easy/numbers__gcd.txt').resolve()],  # TODO hardcode single path to make this easy to run
        "corpus_filepath": Path("environment/corpus/10k_corpus.txt").resolve(),
        "num_problems_per_module": 10 ** 7,
        # data used for validation
        "p_val": 0.2,
    }
    args = parser.parse_args()
    ray.init(num_cpus=args.num_cpus or None)
    ModelCatalog.register_custom_model("keras_model", MyKerasModel)

    # Tests https://github.com/ray-project/ray/issues/7293
    def check_has_custom_metric(result):
        r = result["result"]["info"]["learner"]
        if "default_policy" in r:
            r = r["default_policy"]
        assert r["model"]["foo"] == 42, result

    extra_config = {}

    tune.run(
        args.run,
        stop={"episode_reward_mean": args.stop},
        config=dict(
            extra_config,
            **{
                "env": MathEnv,
                "env_config": env_config,
                # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
                "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
                "callbacks": {
                    "on_train_result": check_has_custom_metric,
                },
                "model": {
                    "keras_model"
                },
                "framework": "tf",
            }))
