import copy
import random
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.special import softmax
from environment.envs import MathEnv
from utils import flatten
from hparams import HParams
hparams = HParams.get_hparams_by_name('rl_math')
# from train import batch_size, model, device, optimizer, max_grad_norm, writer


def get_logdir():
    return f'logs-{hparams.run.name}'


def init_trajectory_data_structures(env):
    '''define data structures to track correct graphs'''
    rewarded_trajectories = {}
    rewarded_trajectory_statistics = {}
    for module_name in env.train.keys():
        for difficulty in env.train[module_name].keys():
            if (module_name, difficulty) not in rewarded_trajectories:
                rewarded_trajectories[(module_name, difficulty)] = []
            if (module_name, difficulty) not in rewarded_trajectory_statistics:
                rewarded_trajectory_statistics[(module_name, difficulty)] = 0
    return rewarded_trajectories, rewarded_trajectory_statistics


def init_envs(env_config, num_environments=10):
    env = MathEnv(env_config)
    envs = [env]
    envs.extend([copy.copy(env) for _ in range(1, num_environments)])
    return envs


def reset_all(envs, rewarded_trajectory_statistics=None, train=True):
    '''if rewarded_trajectory_statistics is not None then select the module_name and difficulty which has been
    least rewarded thus far, else select module_name and difficulty randomly.'''
    envs_info = []
    obs_batch = []
    for env in envs:
        if rewarded_trajectory_statistics is not None:
            module_name, difficulty = min(rewarded_trajectory_statistics, key=rewarded_trajectory_statistics.get)
            obs, info = env.reset_by_module_and_difficulty(module_name, difficulty, train=train)
        else:
            obs, info = env.reset(train=train)
        envs_info.append({'question': info['raw_observation'],
                          'trajectory': [(obs, None, None, None, info)],
                          'module_name': env.module_name,
                          'difficulty': env.difficulty,
                          'module_difficulty_index': env.module_difficulty_index})
        obs_batch.append(np.expand_dims(obs, 0))
    obs_batch = np.concatenate(obs_batch)
    return obs_batch, envs_info


def step_all(envs, action_batch):
    step_batch = list()
    obs_batch = list()
    for env, action in zip(envs, action_batch):
        step = env.step(action)
        (obs, reward, done, info) = step
        step_batch.append(step)
        obs_batch.append(np.expand_dims(obs, 0))
    obs_batch = np.concatenate(obs_batch)
    return obs_batch, step_batch


def get_action_batch(obs_batch, envs, model=None):
    # get model output
    if model:
        obs_batch = torch.from_numpy(obs_batch.astype(np.int64))
        output_batch = model(obs_batch.to(model.device)).detach().cpu().numpy()
        model_type = hparams.model.model_type
    else:
        output_batch = np.random.uniform(size=(len(obs_batch),len(envs[0].actions)))
        model_type = 'policy'
    if model_type == 'policy':
        output_batch = softmax(output_batch, axis=1)
    actions = []
    for i, env in enumerate(envs):
        # mask the corresponding model output
        masked_output = env.mask_invalid_types(output_batch[i])
        # assert np.sum(masked_output) != 0
        if model_type == 'policy':
            # normalize and sample
            # TODO: remove conditional if assert never triggered?
            masked_policy_vector = masked_output if np.sum(masked_output) != 0 else np.ones(
                len(masked_output), dtype=np.int64)
            masked_normed_policy_vector = masked_policy_vector / np.sum(masked_policy_vector)
            action_index = np.random.choice(env.action_indices, p=masked_normed_policy_vector)
        elif model_type == 'value':
            eps_ = random.random()
            if eps_ < model.epsilon:
                # take random action
                # TODO: remove conditional if assert never triggered?
                available_actions = [i for i in env.action_indices if masked_output[i] != 0] if np.sum(masked_output) != 0 else np.ones(
                    len(masked_output), dtype=np.int64)
                action_index = random.choice(available_actions)
            else:
                action_index = np.argmax(masked_output)
        actions.append(action_index)
    return actions


def update_trajectory_data_structures(env_info, rewarded_trajectories, rewarded_trajectory_statistics):
    module_name = env_info['module_name']
    difficulty = env_info['difficulty']
    trajectory = env_info['trajectory']
    reward = trajectory[-1][2]
    if reward == 1:
        rewarded_trajectories[(module_name, difficulty)].append(trajectory)
        rewarded_trajectory_statistics[(module_name, difficulty)] += 1


def reset_environment(env, train=True):
    obs, info = env.reset(train=train)
    return obs, {'question': info['raw_observation'],
                 'trajectory': [(obs, None, None, None, None)],
                 'module_name': env.module_name,
                 'difficulty': env.difficulty,
                 'module_difficulty_index': env.module_difficulty_index}


def reset_environment_with_least_rewarded_problem_type(env, rewarded_trajectory_statistics, train=True):
    module_name, difficulty = min(rewarded_trajectory_statistics, key=rewarded_trajectory_statistics.get)
    obs, info = env.reset_by_module_and_difficulty(module_name, difficulty, train=train)
    return obs, {'question': info['raw_observation'],
                 'trajectory': [(obs, None, None, None, None)],
                 'module_name': module_name,
                 'difficulty': difficulty,
                 'module_difficulty_index': env.module_difficulty_index}


def align_trajectory(raw_trajectory):
    states = [state for state, _, _, _, _ in raw_trajectory[:-1]]
    everything_else = [(next_state, action, reward, done) for next_state, action, reward, done, _ in raw_trajectory[1:]]
    aligned_trajectory = [(state, action, reward, next_state, done)
                         for state, (next_state, action, reward, done)
                         in zip(states, everything_else)]
    return aligned_trajectory


def inspect_performance(trajectories, rewarded_trajectory_statistics):
    for module_name, difficulty in trajectories.keys():
        if len(trajectories[(module_name,difficulty)]) > 0:
            percentage_correct = rewarded_trajectory_statistics[(module_name,difficulty)] / len(trajectories[(module_name,difficulty)]) * 100
            print(f"{module_name}@{difficulty}: {rewarded_trajectory_statistics[(module_name,difficulty)]} / {len(trajectories[(module_name,difficulty)])} = {round(percentage_correct, 5)}%")


# make function to compute action distribution
def get_policy(model, obs):
    from torch.distributions.categorical import Categorical
    logits = model(obs)
    return Categorical(logits=logits)


# make loss function whose gradient, for the right data, is policy gradient
def vpg_loss(model, obs, act, weights):
    logp = get_policy(model, obs).log_prob(act)
    return -(logp * weights).mean()


def vpg_step(model, state_batch, action_batch, reward_batch):
    # take a single policy gradient update step
    model.optimizer.zero_grad()
    batch_loss = vpg_loss(model=model, obs=state_batch, act=action_batch, weights=reward_batch)
    batch_loss.backward()
    # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), model.max_grad_norm)
    model.optimizer.step()
    return batch_loss


def dqn_step(model, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
    # Take a single deep Q learning update step
    targets = reward_batch + (1 - done_batch) * hparams.train.gamma * torch.max(model(next_state_batch), dim=1)[0]
    model.optimizer.zero_grad()
    batch_output = model(state_batch)
    batch_output = batch_output.gather(1, action_batch.view(-1,1)).squeeze()
    batch_loss = torch.nn.MSELoss()(batch_output, targets)
    batch_loss.backward()
    # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), model.max_grad_norm)
    model.optimizer.step()
    return batch_loss


class StepDataset(torch.utils.data.Dataset):
    """Step Dataset"""

    def __init__(self, trajectory_buffer, device):
        self.step_buffer = flatten(trajectory_buffer)
        self.device = device

    def __len__(self):
        return len(self.step_buffer)

    def __getitem__(self, idx):
        # return only the (state, action, reward)?
        state, action, reward, next_state, done = self.step_buffer[idx]
        state = torch.from_numpy(state.astype(np.int64)).to(self.device)
        action = torch.from_numpy(np.array(action, dtype=np.int64)).to(self.device)
        reward = torch.from_numpy(np.array(reward, dtype=np.int64)).to(self.device)
        next_state = torch.from_numpy(next_state.astype(np.int64)).to(self.device)
        done = torch.from_numpy(np.array(done, dtype=np.int64)).to(self.device)
        return state, action, reward, next_state, done


def train_on_buffer(model, data_loader, n_batches, writer, current_batch_i):
    model.train()
    for i, (state_batch, action_batch, reward_batch, next_state_batch, done_batch) in enumerate(data_loader):
        batch_loss = dqn_step(model, state_batch, action_batch, reward_batch, next_state_batch, done_batch)
        # batch_loss = vpg_step(model, state_batch, action_batch, reward_batch)
        writer.add_scalar('Train/loss', batch_loss, current_batch_i)
        # writer.add_scalar('Train/gradients', grad_norm, current_batch_i)
        current_batch_i += 1
        if i >= n_batches:
            break
    return current_batch_i


def run_eval(model, envs, writer, batch_i, n_required_validation_episodes):
    model.eval()
    logdir = get_logdir()
    total_reward = {}  # key: (module_name, difficulty) val: dict[key: n_completed_episodes or tot_reward]
    n_completed_validation_episodes = 0
    obs_batch, envs_info = reset_all(envs, train=False)
    while True:
        # take a step in each environment in "parallel"
        with torch.no_grad():
            action_batch = get_action_batch(obs_batch, envs, model=model)
        obs_batch, step_batch = step_all(envs, action_batch)
        # for each environment process the most recent step
        for env_i, ((obs, reward, done, info), action) in enumerate(zip(step_batch, action_batch)):
            envs_info[env_i]['trajectory'].append((obs.astype(np.int16), action, reward, done, info))
            # if episode is complete, check if trajectory should be kept in buffer and reset environment
            if done:
                k = (envs[env_i].module_name, envs[env_i].difficulty)
                with open(f'{logdir}/validation_graphs_{k[0]}_{k[1]}.txt', 'a') as f:
                    f.write(f"{info['raw_observation']} = {envs[env_i].compute_graph.eval()}, reward: {reward}\n")
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
        mean_val_reward_per_module = total_reward[k]["tot_reward"] / total_reward[k]["n_completed_validation_episodes"]
        all_modules_reward += total_reward[k]["tot_reward"]
        writer.add_scalar(f'Val/{k[0]}_{k[1]}_reward', mean_val_reward_per_module, batch_i)

    mean_val_reward = all_modules_reward / n_completed_validation_episodes
    # check whether LR should be annealed via ReduceLROnPlateau
    model.scheduler.step(mean_val_reward)
    writer.add_scalar('Train/lr', model.optimizer.param_groups[0]['lr'], batch_i)
    writer.add_scalar('Val/tot_reward', mean_val_reward, batch_i)
    print(f'{batch_i} batches completed, mean validation reward: {mean_val_reward}')
    writer.close()


def visualize_buffer(buffer, env):
    states = [state for state, _, _ in buffer]
    actions = [action for _, action, _ in buffer]
    decoded_states = [env.decode(state) for state in states]
    for d, a in zip(decoded_states, actions):
        print(d, a)
    print()


def fill_buffer(model, envs, buffer_threshold, positive_to_negative_ratio, rewarded_trajectories,
                rewarded_trajectory_statistics, verbose=False, mode='positive_only', max_num_steps=1000):
    '''
    :param model:
    :param envs:
    :param buffer_threshold:
    :param positive_to_negative_ratio:
    :param rewarded_trajectories:
    :param rewarded_trajectory_statistics:
    :param verbose:
    :param mode: can be 'positive_only' and 'balanced'
    :return:
    '''
    # reset all environments
    cached_steps = 0
    trajectory_buffer = []
    buffer_positives = 1
    buffer_negatives = 1  # init to 1 to prevent division by zero
    obs_batch, envs_info = reset_all(envs, rewarded_trajectory_statistics=rewarded_trajectory_statistics, train=True)
    # take steps in all environments num_parallel_steps times
    for _ in range(max_num_steps):
        # take a step in each environment in "parallel"
        action_batch = get_action_batch(obs_batch, envs, model=model)
        obs_batch, step_batch = step_all(envs, action_batch)
        # for each environment process the most recent step
        for env_i, ((obs, reward, done, info), action) in enumerate(zip(step_batch, action_batch)):
            envs_info[env_i]['trajectory'].append((obs.astype(np.int16), action, reward, done, info))
            # if episode is complete, check if trajectory should be kept in trajectory_buffer and reset environment
            if done:
                update_trajectory_data_structures(envs_info[env_i], rewarded_trajectories, rewarded_trajectory_statistics)
                with open(f'{get_logdir()}/training_graphs.txt', 'a') as f:
                    f.write(f"{info['raw_observation']} = {envs[env_i].compute_graph.eval()}\n")
                if reward == 1:
                    # cache trajectory
                    pass
                if reward == 1 and verbose:
                    print(f"{info['raw_observation']} = {envs[env_i].compute_graph.eval()}")
                if (mode == 'positive_only' and reward == 1) or \
                   (mode == 'balanced' and buffer_positives / buffer_negatives <= positive_to_negative_ratio and \
                        reward == 1):
                    aligned_trajectory = align_trajectory(envs_info[env_i]['trajectory'])
                    trajectory_buffer.append(aligned_trajectory)
                    cached_steps += len(aligned_trajectory)
                    buffer_positives += 1
                elif mode == 'balanced' and buffer_positives / buffer_negatives > positive_to_negative_ratio and \
                        reward == -1:
                    aligned_trajectory = align_trajectory(envs_info[env_i]['trajectory'])
                    trajectory_buffer.append(aligned_trajectory)
                    cached_steps += len(aligned_trajectory)
                    buffer_negatives += 1
                obs_batch[env_i], envs_info[env_i] = \
                    reset_environment_with_least_rewarded_problem_type(envs[env_i], rewarded_trajectory_statistics,
                                                                       train=True)
                # # append first state of trajectory after reset
                # info_dict = {'raw_observation': envs_info[env_i]['question']}
                # envs_info[env_i]['trajectory'].append((obs_batch[env_i].astype(np.int16), None, None, None, info_dict))
        if cached_steps > buffer_threshold:
            break
    return trajectory_buffer


def load_buffer(trajectories_filepath):
    from utils import read_pickle
    buffer = list()
    trajectories = read_pickle(trajectories_filepath)
    for module_name, difficulty in trajectories:
        trajectories_ = trajectories[(module_name, difficulty)]
        for trajectory in trajectories_:
            reward = trajectory[-1][2]
            processed_trajectory = align_trajectory(trajectory, reward)
            buffer.extend(processed_trajectory)
    return buffer