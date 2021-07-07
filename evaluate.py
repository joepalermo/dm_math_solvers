from hparams import HParams
hparams = HParams('.', hparams_filename='hparams', name='rl_math')
# hparams = HParams('.', hparams_filename='hparams', name='rl_math', ask_before_deletion=False)
import torch
from torch.utils.tensorboard import SummaryWriter
from modelling.train_utils import init_trajectory_data_structures, init_envs, get_logdir, run_test
from modelling.transformer_encoder import TransformerEncoderModel
import numpy as np
import os

# basic setup and checks
torch.manual_seed(hparams.run.seed)
np.random.seed(seed=hparams.run.seed)
device = torch.device(f'cuda:{hparams.run.gpu_id}' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(log_dir=get_logdir())
logging_batches_filepath = os.path.join(get_logdir(), hparams.run.logging_batches_filename)
logging_q_values_filepath = os.path.join(get_logdir(), hparams.run.logging_q_values_filename)

# initialize all environments
envs = init_envs(hparams.env)
trajectory_statistics = init_trajectory_data_structures(envs[0])

# init models
num_question_tokens = hparams.env.question_vocab_size + 1
num_outputs = envs[0].num_actions
question_padding_token = envs[0].question_padding_token
action_padding_token = envs[0].action_padding_token


q1 = TransformerEncoderModel(num_question_tokens=num_question_tokens, num_outputs=num_outputs,
                             question_padding_token=question_padding_token, action_padding_token=action_padding_token,
                             device=device)
q2 = TransformerEncoderModel(num_question_tokens=num_question_tokens, num_outputs=num_outputs,
                             question_padding_token=question_padding_token, action_padding_token=action_padding_token,
                             device=device)

batch_i = last_fill_buffer_batch_i = last_eval_batch_i = last_target_network_update_batch_i = 0

# Load checkpoints that must exist
if os.path.isfile(os.path.join(get_logdir(), hparams.model.q1_file)):
    print("Checkpoint found, loading checkpoints")
    q1_checkpoint = torch.load(os.path.join(get_logdir(), hparams.model.q1_file))
    q2_checkpoint = torch.load(os.path.join(get_logdir(), hparams.model.q2_file))
    q1.load_state_dict(q1_checkpoint['model_state_dict'])
    q2.load_state_dict(q2_checkpoint['model_state_dict'])
    q1.optimizer.load_state_dict(q1_checkpoint['optimizer_state_dict'])
    q2.optimizer.load_state_dict(q2_checkpoint['optimizer_state_dict'])
    batch_i = last_fill_buffer_batch_i = last_eval_batch_i = last_target_network_update_batch_i = q1_checkpoint['batch']
else:
    raise ValueError("no trained models found")

# Run evaluation on test set
for module in hparams.env.selected_filenames:
    run_test(q1, envs, module[:-4])
