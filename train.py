from hparams import HParams
# hparams = HParams('.', hparams_filename='hparams', name='rl_math')
hparams = HParams('.', hparams_filename='hparams', name='rl_math', ask_before_deletion=False)
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from modelling.train_utils import init_trajectory_data_structures, init_envs, train_on_buffer, run_eval, fill_buffer, \
    load_buffer, get_logdir, visualize_buffer, StepDataset
from modelling.transformer_encoder import TransformerEncoderModel

# basic setup and checks
device = torch.device(f'cuda:{hparams.run.gpu_id}' if torch.cuda.is_available() else 'cpu')
assert hparams.train.mode == 'positive_only' or hparams.train.mode == 'balanced'
writer = SummaryWriter(log_dir=get_logdir())

# initialize all environments
envs = init_envs(hparams.env)
rewarded_trajectories, rewarded_trajectory_statistics = init_trajectory_data_structures(envs[0])

# set dependent params
ntoken = hparams.env.vocab_size + 1
num_outputs = len(envs[0].actions)

# load or init model
if hparams.model.model_filepath is not None:
    model = torch.load(hparams.model.model_filepath)
else:
    dummy_model = None
    model = TransformerEncoderModel(ntoken=ntoken, num_outputs=num_outputs, device=device)

# bootstrap
# buffer = load_buffer('mathematics_dataset-v1.0/differentiate_50_buffers.pkl')
# batch_i = train_on_buffer(model, buffer, writer, batch_i, batches_per_train)
# run_eval(model, envs, writer, batch_i, 100)

# TODO: store s_t+1? or handle that in extraction? (for DQN)

# training loop
batch_i = 0
last_eval_batch_i = 0
replay_buffer = []
for buffer_i in tqdm(range(hparams.train.num_buffers)):
    trajectory_buffer = fill_buffer(dummy_model, envs, hparams.train.buffer_threshold, hparams.train.positive_to_negative_ratio, rewarded_trajectories,
                         rewarded_trajectory_statistics, mode=hparams.train.mode, max_num_steps=hparams.train.fill_buffer_max_steps, verbose=True)
    # trajectory_buffer = fill_buffer(model, envs, hparams.train.buffer_threshold, hparams.train.positive_to_negative_ratio, rewarded_trajectories,
    #                      rewarded_trajectory_statistics, mode=hparams.train.mode, max_num_steps=hparams.train.fill_buffer_max_steps)
    # construct dataset
    if hparams.train.use_replay_buffer:
        replay_buffer.extend(trajectory_buffer)
        step_dataset = StepDataset(replay_buffer, model.device)
    else:
        step_dataset = StepDataset(trajectory_buffer, model.device)
    # construct data loader
    data_loader = DataLoader(step_dataset, batch_size=model.batch_size, shuffle=True)
    # train
    batches_in_dataset = len(step_dataset) // model.batch_size
    batches_to_train = min(batches_in_dataset, hparams.train.batches_per_train)
    batch_i = train_on_buffer(model, data_loader, batches_to_train, writer, batch_i)
    # eval
    if batch_i - last_eval_batch_i >= hparams.train.batches_per_eval:
        last_eval_batch_i = batch_i
        run_eval(model, envs, writer, batch_i, hparams.train.n_required_validation_episodes)

# from utils import write_pickle
# write_pickle('mathematics_dataset-v1.0/trajectories.pkl', rewarded_trajectories)
