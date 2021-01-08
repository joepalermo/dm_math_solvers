import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from modelling.train_utils import init_trajectory_data_structures, init_envs, train_on_buffer, run_eval, fill_buffer, \
    load_buffer
from modelling.transformer_encoder import TransformerEncoderModel

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(0)

selected_filenames = [
    #'numbers__is_factor.txt',  # checked
    # 'numbers__is_prime.txt',  # checked
    #'numbers__list_prime_factors.txt',  # checked
    'calculus__differentiate.txt',  # checked
    #'polynomials__evaluate.txt',  # checked
    # 'numbers__div_remainder.txt',  # checked
    # 'numbers__gcd.txt',
    # 'numbers__lcm.txt',
    # 'algebra__linear_1d.txt',
    # 'algebra__polynomial_roots.txt',
    # 'algebra__linear_2d.txt',
    # 'algebra__linear_1d_composed.txt',
    # 'algebra__linear_2d_composed.txt',
    # 'algebra__polynomial_roots_composed.txt',
    # 'calculus__differentiate_composed.txt',
    # 'numbers__div_remainder_composed.txt',
    # 'numbers__gcd_composed.txt',
    # 'numbers__is_factor_composed.txt',
    # 'numbers__is_prime_composed.txt',
    # 'numbers__lcm_composed.txt',
    # 'numbers__list_prime_factors_composed.txt',
    # 'polynomials__evaluate_compose.txt'
    # 'polynomials__compose.txt',
]

# define and init environment
filepaths = [
    f"mathematics_dataset-v1.0/train-easy/{fn}" for fn in selected_filenames if 'composed' not in fn
]

env_config = {
    "problem_filepaths": filepaths,
    "corpus_filepath": str(Path("environment/corpus/10k_corpus.txt").resolve()),
    "num_problems_per_module": 10 ** 6,
    "validation_percentage": 0.2,
    "max_sequence_length": 800,
    "vocab_size": 200,
    "max_difficulty": 0  # i.e. uncomposed only
}

# define search parameters
verbose = False
num_steps = 50000
num_environments = 32

# initialize all environments
envs = init_envs(env_config, num_environments)
rewarded_trajectories, rewarded_trajectory_statistics = init_trajectory_data_structures(envs[0])

# architecture params
ntoken = env_config['vocab_size'] + 1
nhead = 8
nhid = 512
nlayers = 1
num_outputs = len(envs[0].actions)
dropout = 0.1

# training params
batch_size = 8
buffer_threshold = batch_size
positive_to_negative_ratio = 1
lr = 0.5
n_required_validation_episodes = 2000
max_grad_norm = 0.05

# load model
load_model = False
if load_model:
    model = torch.load('modelling/models/model.pt')
else:
    dummy_model = None
    model = TransformerEncoderModel(ntoken=ntoken, nhead=nhead, nhid=nhid, nlayers=nlayers, num_outputs=num_outputs,
                dropout=dropout, device=device, lr=lr, max_grad_norm=max_grad_norm, batch_size=batch_size)

writer = SummaryWriter(comment="calculus__differentiate")

num_buffers = 1000000000
fill_buffer_max_steps = 1000
last_eval_batch_i = 0
batches_per_eval = 10
batches_per_train = 10
batch_i = 0
mode = 'positive_only'
assert mode == 'positive_only' or mode == 'balanced'

# bootstrap
# buffer = load_buffer('mathematics_dataset-v1.0/differentiate_50_buffers.pkl')
# batch_i = train_on_buffer(model, buffer, writer, batch_i, batches_per_train)
# run_eval(model, envs, writer, batch_i, 100)

# training loop
replay_buffer = []
for buffer_i in tqdm(range(num_buffers)):
    # buffer = fill_buffer(dummy_model, envs, buffer_threshold, positive_to_negative_ratio, rewarded_trajectories,
    #                      rewarded_trajectory_statistics, mode=mode, max_num_steps=fill_buffer_max_steps)
    buffer = fill_buffer(model, envs, buffer_threshold, positive_to_negative_ratio, rewarded_trajectories,
                         rewarded_trajectory_statistics, mode=mode, max_num_steps=fill_buffer_max_steps)
    replay_buffer.extend(buffer)
    batch_i = train_on_buffer(model, replay_buffer, writer, batch_i, batches_per_train)
    # eval
    if batch_i - last_eval_batch_i >= batches_per_eval:
        last_eval_batch_i = batch_i
        run_eval(model, envs, writer, batch_i, n_required_validation_episodes)

# from utils import write_pickle
# write_pickle('mathematics_dataset-v1.0/trajectories.pkl', rewarded_trajectories)
