from hparams import HParams

hparams = HParams('.', hparams_filename='hparams', name='rl_math')

import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from modelling.train_utils import init_trajectory_data_structures, init_envs, train_on_buffer, run_eval, fill_buffer, \
    load_buffer, get_logdir
from modelling.transformer_encoder import TransformerEncoderModel

logdir = get_logdir()

device = torch.device(f'cuda:{hparams.run.gpu_id}' if torch.cuda.is_available() else 'cpu')

# define and init environment
filepaths = [
    f"mathematics_dataset-v1.0/train-easy/{fn}" for fn in hparams.env.selected_filenames if 'composed' not in fn
]

env_config = {
    "problem_filepaths": filepaths,
    "corpus_filepath": str(Path("environment/corpus/20k_question_corpus.txt").resolve()),
    "num_problems_per_module": hparams.env.num_problems_per_module,
    "validation_percentage": hparams.env.validation_percentage,
    "max_sequence_length": hparams.env.max_sequence_length,
    "vocab_size": hparams.env.vocab_size,
    "univariate_differentiation": hparams.env.univariate_differentiation,
    "max_difficulty": hparams.env.max_difficulty  # i.e. uncomposed only
}

# initialize all environments
envs = init_envs(env_config, hparams.env.num_environments)
rewarded_trajectories, rewarded_trajectory_statistics = init_trajectory_data_structures(envs[0])

# architecture params
ntoken = env_config['vocab_size'] + 1
num_outputs = len(envs[0].actions)

# training params
buffer_threshold = hparams.train.batch_size

# load model
load_model = False
if load_model:
    model = torch.load('modelling/models/model.pt')
else:
    dummy_model = None
    model = TransformerEncoderModel(ntoken=ntoken, nhead=hparams.model.nhead, nhid=hparams.model.nhid, nlayers=hparams.model.nlayers, num_outputs=num_outputs,
                dropout=hparams.model.dropout, device=device, lr=hparams.train.lr, max_grad_norm=hparams.train.max_grad_norm, batch_size=hparams.train.batch_size)

writer = SummaryWriter(log_dir=logdir)

mode = hparams.train.mode
assert mode == 'positive_only' or mode == 'balanced'

# bootstrap
# buffer = load_buffer('mathematics_dataset-v1.0/differentiate_50_buffers.pkl')
# batch_i = train_on_buffer(model, buffer, writer, batch_i, batches_per_train)
# run_eval(model, envs, writer, batch_i, 100)
# training loop
batch_i = 0
last_eval_batch_i = 0
replay_buffer = []
for buffer_i in tqdm(range(hparams.train.num_buffers)):
    # buffer = fill_buffer(dummy_model, envs, buffer_threshold, positive_to_negative_ratio, rewarded_trajectories,
    #                      rewarded_trajectory_statistics, mode=mode, max_num_steps=fill_buffer_max_steps)
    buffer = fill_buffer(model, envs, buffer_threshold, hparams.train.positive_to_negative_ratio, rewarded_trajectories,
                         rewarded_trajectory_statistics, mode=mode, max_num_steps=hparams.train.fill_buffer_max_steps)
    replay_buffer.extend(buffer)
    batch_i = train_on_buffer(model, replay_buffer, writer, batch_i, hparams.train.batches_per_train)
    # eval
    if batch_i - last_eval_batch_i >= hparams.train.batches_per_eval:
        last_eval_batch_i = batch_i
        run_eval(model, envs, writer, batch_i, hparams.train.n_required_validation_episodes)

# from utils import write_pickle
# write_pickle('mathematics_dataset-v1.0/trajectories.pkl', rewarded_trajectories)
