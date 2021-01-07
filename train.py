import random
import numpy as np
import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from modelling.train_utils import init_trajectory_data_structures, init_envs, reset_all, step_all, get_action_batch, \
    update_rewarded_trajectory_statistics, reset_environment, reset_environment_with_least_rewarded_problem_type, \
    extract_buffer_trajectory
from modelling.transformer_encoder import TransformerEncoderModel

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(0)

selected_filenames = [
    'numbers__is_factor.txt',
    # 'numbers__is_prime.txt',
    # 'numbers__list_prime_factors.txt',
    # 'calculus__differentiate.txt',
    # 'polynomials__evaluate.txt',
    # 'numbers__div_remainder.txt',
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
    "max_sequence_length": 200,
    "vocab_size": 200,
    "max_difficulty": 0  # i.e. uncomposed only
}

# define search parameters
verbose = False
num_steps = 5000000
num_environments = 32

# initialize all environments
envs = init_envs(env_config, num_environments)
rewarded_trajectory_statistics = init_trajectory_data_structures(envs[0])

# architecture params
ntoken = env_config['vocab_size'] + 1
nhead = 4
nhid = 128
nlayers = 1
num_outputs = len(envs[0].actions)
dropout = 0.1

# training params
batch_size = 32
buffer_threshold = batch_size
positive_to_negative_ratio = 1
lr = 0.05
n_required_validation_episodes = 1000
max_grad_norm = 0.5

# load model
load_model = False
if load_model:
    model = torch.load('modelling/models/model.pt')
else:
    dummy_model = None
    model = TransformerEncoderModel(ntoken=ntoken, nhead=nhead, nhid=nhid, nlayers=nlayers, num_outputs=num_outputs,
                                dropout=dropout, device=device)
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=20,
                                                       threshold=0.001, threshold_mode='rel', cooldown=0,
                                                       min_lr=0, eps=1e-08, verbose=False)
writer = SummaryWriter()

# reset all environments
buffer = []
buffer_positives = 1
buffer_negatives = 1  # init to 1 to prevent division by zero
total_batches = 0
obs_batch, envs_info = reset_all(envs, rewarded_trajectory_statistics=rewarded_trajectory_statistics, train=True)
# take steps in all environments num_parallel_steps times
num_parallel_steps = num_steps // num_environments
for parallel_step_i in tqdm(range(num_parallel_steps)):
    # take a step in each environment in "parallel"
    action_batch = get_action_batch(obs_batch, envs, device=device, model=model)
    obs_batch, step_batch = step_all(envs, action_batch)
    # for each environment process the most recent step
    for env_i, ((obs, reward, done, info), action) in enumerate(zip(step_batch, action_batch)):
        envs_info[env_i]['trajectory'].append((obs.astype(np.int16), action, reward, done, info))
        # if episode is complete, check if trajectory should be kept in buffer and reset environment
        if done:
            update_rewarded_trajectory_statistics(envs_info[env_i], rewarded_trajectory_statistics)
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
    if len(buffer) > buffer_threshold:
        # n_batches = math.floor(len(buffer) / batch_size)
        n_batches = 1
        total_batches += n_batches
        random.shuffle(buffer)
        model.train()
        for batch_i in range(n_batches):
            batch = buffer[batch_i * batch_size : (batch_i + 1) * batch_size]
            state_batch = torch.from_numpy(np.concatenate([np.expand_dims(step[0], 0) for step in batch]).astype(np.int64)).to(device)
            action_batch = torch.from_numpy(np.concatenate([np.expand_dims(step[1], 0) for step in batch]).astype(np.int64)).to(device)
            reward_batch = torch.from_numpy(np.concatenate([np.expand_dims(step[2], 0) for step in batch]).astype(np.int64)).to(device)
            batch_logits = model(state_batch)
            batch_probs = torch.softmax(batch_logits, axis=1)
            # loss is given by -mean(log(model(a=a_t|s_t)) * R_t)
            loss = -torch.mean(torch.log(batch_probs[:, action_batch]) * reward_batch)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            writer.add_scalar('Loss/train', loss, total_batches)
            writer.add_scalar('Gradients/train', grad_norm, total_batches)
        # reset buffer
        buffer = []

        # for annealing LR via StepLR
        # if total_batches % 10 == 0:
        #     scheduler.step()

        if total_batches % 1 == 0:
            model.eval()
            total_reward = {}  # key: (module_name, difficulty) val: dict[key: n_completed_episodes or tot_reward]
            n_completed_validation_episodes = 0
            obs_batch, envs_info = reset_all(envs, train=False)
            # take steps in all environments num_parallel_steps times
            num_parallel_steps = num_steps // num_environments
            for parallel_step_i in range(num_parallel_steps):
                # take a step in each environment in "parallel"
                with torch.no_grad():
                    action_batch = get_action_batch(obs_batch, envs, device=device, model=model)
                obs_batch, step_batch = step_all(envs, action_batch)
                # for each environment process the most recent step
                for env_i, ((obs, reward, done, info), action) in enumerate(zip(step_batch, action_batch)):
                    envs_info[env_i]['trajectory'].append((obs.astype(np.int16), action, reward, done, info))
                    # if episode is complete, check if trajectory should be kept in buffer and reset environment
                    if done:
                        with open('modelling/validation_graphs.txt', 'a') as f:
                            f.write(f"{info['raw_observation']} = {envs[env_i].compute_graph.eval()}, reward: {reward}\n")
                        k = (envs[env_i].module_name, envs[env_i].difficulty)
                        if k in total_reward:
                            total_reward[k]["n_completed_validation_episodes"] += 1
                            total_reward[k]["tot_reward"] += reward
                        else:
                            total_reward[k] = {}
                            total_reward[k]["n_completed_validation_episodes"] = 1
                            total_reward[k]["tot_reward"] = reward
                        n_completed_validation_episodes += 1

                        obs_batch[env_i], envs_info[env_i] = reset_environment(envs[env_i], train=False)
                if n_completed_validation_episodes > n_required_validation_episodes:
                    break
            all_modules_reward = 0
            for k in total_reward.keys():
                mean_val_reward = total_reward[k]["tot_reward"] / total_reward[k]["n_completed_validation_episodes"]
                all_modules_reward += total_reward[k]["tot_reward"]
                writer.add_scalar(f'Val/{k[0]}_{k[1]}_reward', mean_val_reward, total_batches)
                # check whether LR should be annealed via ReduceLROnPlateau
                scheduler.step(mean_val_reward)

            mean_val_reward = all_modules_reward / n_completed_validation_episodes
            writer.add_scalar('Val/tot_reward', mean_val_reward, total_batches)
            print(f'{total_batches} batches completed, mean validation reward: {mean_val_reward}')
            writer.close()