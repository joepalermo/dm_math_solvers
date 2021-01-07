import numpy as np
import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from modelling.train_utils import init_trajectory_data_structures, init_envs, reset_all, step_all, get_action_batch, \
    update_trajectory_data_structures, reset_environment, reset_environment_with_least_rewarded_problem_type, \
    extract_buffer_trajectory, train_on_buffer, run_eval
from modelling.transformer_encoder import TransformerEncoderModel

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(0)

selected_filenames = [
    'numbers__is_factor.txt',  # checked
    # 'numbers__is_prime.txt',  # checked
    # 'numbers__list_prime_factors.txt',  # checked
    # 'calculus__differentiate.txt',  # checked
    # 'polynomials__evaluate.txt',  # checked
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
    "max_sequence_length": 400,
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
nhead = 4
nhid = 128
nlayers = 1
num_outputs = len(envs[0].actions)
dropout = 0.1

# training params
data_collection_only = False
batch_size = 8
buffer_threshold = batch_size
positive_to_negative_ratio = 1
lr = 0.05
batches_per_eval = 1
n_required_validation_episodes = 1000
max_grad_norm = 0.5

# load model
load_model = False
if load_model:
    model = torch.load('modelling/models/model.pt')
else:
    dummy_model = None
    model = TransformerEncoderModel(ntoken=ntoken, nhead=nhead, nhid=nhid, nlayers=nlayers, num_outputs=num_outputs,
                dropout=dropout, device=device, lr=lr, max_grad_norm=max_grad_norm, batch_size=batch_size)

writer = SummaryWriter()

# reset all environments
buffer = []
buffer_positives = 1
buffer_negatives = 1  # init to 1 to prevent division by zero
batch_i = last_eval_batch_i = 0
obs_batch, envs_info = reset_all(envs, rewarded_trajectory_statistics=rewarded_trajectory_statistics, train=True)
# take steps in all environments num_parallel_steps times
num_parallel_steps = num_steps // num_environments
for parallel_step_i in tqdm(range(num_parallel_steps)):
    # take a step in each environment in "parallel"
    action_batch = get_action_batch(obs_batch, envs, model=dummy_model)
    obs_batch, step_batch = step_all(envs, action_batch)
    # for each environment process the most recent step
    for env_i, ((obs, reward, done, info), action) in enumerate(zip(step_batch, action_batch)):
        envs_info[env_i]['trajectory'].append((obs.astype(np.int16), action, reward, done, info))
        # if episode is complete, check if trajectory should be kept in buffer and reset environment
        if done:
            update_trajectory_data_structures(envs_info[env_i], rewarded_trajectories, rewarded_trajectory_statistics)
            with open('modelling/training_graphs.txt', 'a') as f:
                f.write(f"{info['raw_observation']} = {envs[env_i].compute_graph.eval()}\n")
            if reward == 1 and verbose:
                print(f"{info['raw_observation']} = {envs[env_i].compute_graph.eval()}")
            if buffer_positives/buffer_negatives <= positive_to_negative_ratio and reward == 1:
                buffer_trajectory = extract_buffer_trajectory(envs_info[env_i]['trajectory'], reward)
                buffer.extend(buffer_trajectory)
                buffer_positives += 1
            elif buffer_positives/buffer_negatives > positive_to_negative_ratio and reward == -1:
                buffer_trajectory = extract_buffer_trajectory(envs_info[env_i]['trajectory'], reward)
                buffer.extend(buffer_trajectory)
                buffer_negatives += 1
            obs_batch[env_i], envs_info[env_i] = \
                reset_environment_with_least_rewarded_problem_type(envs[env_i], rewarded_trajectory_statistics,
                                                                   train=True)

    # when enough steps have been put into buffer, construct training batches and fit
    if len(buffer) > buffer_threshold and not data_collection_only:
        # train
        train_on_buffer(model, buffer, writer, batch_i)

        # reset buffer
        buffer = []

        # eval
        if batch_i - last_eval_batch_i >= batches_per_eval:
            last_eval_batch_i = batch_i
            run_eval(model, envs, writer, batch_i, n_required_validation_episodes)

from utils import write_pickle
write_pickle('mathematics_dataset-v1.0/trajectories.pkl', rewarded_trajectories)
