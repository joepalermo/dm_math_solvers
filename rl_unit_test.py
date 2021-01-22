import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from hparams import HParams
hparams = HParams('.', hparams_filename='hparams', name='rl_math', ask_before_deletion=False)
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import flatten
from modelling.train_utils import train_on_buffer, get_logdir
device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(log_dir=get_logdir())
import numpy as np


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
        state = torch.from_numpy(state.astype(np.float32)).to(self.device)
        action = torch.from_numpy(np.array(action, dtype=np.int64)).to(self.device)
        reward = torch.from_numpy(np.array(reward, dtype=np.float32)).to(self.device)
        next_state = torch.from_numpy(next_state.astype(np.float32)).to(self.device)
        done = torch.from_numpy(np.array(done, dtype=np.int64)).to(self.device)
        return state, action, reward, next_state, done


def generate_trajectories(epsilon, num_episodes=1, episode_timeout=100):
    trajectories = list()
    env = gym.make('CartPole-v0')
    for i_episode in range(num_episodes):
        state = env.reset()
        trajectory = []
        for t in range(episode_timeout):
            #env.render()
            torch_state = torch.from_numpy(state.astype(np.float32)).to(model.device)
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = torch.argmax(model(torch_state)).item()
            next_state, reward, done, info = env.step(action)
            trajectory.append((state, action, reward, next_state, done))
            state = next_state
            if done:
                #print("Episode finished after {} timesteps".format(t+1))
                trajectories.append(trajectory)
                break
    env.close()
    return trajectories

def run_eval(model, num_episodes=200, episode_timeout=100):
    env = gym.make('CartPole-v0')
    rewards = list()
    for i_episode in range(num_episodes):
        tot_reward = 0
        state = env.reset()
        for t in range(episode_timeout):
            #env.render()
            state = torch.from_numpy(state.astype(np.float32)).to(model.device)
            action = torch.argmax(model(state))
            state, reward, done, info = env.step(action.item())
            tot_reward += reward
            if done:
                #print("Episode finished after {} timesteps".format(t+1))
                break
        rewards.append(tot_reward)
    env.close()
    print("Mean Reward per episode:", sum(rewards)/len(rewards))


class MLP(nn.Module):
    def __init__(self, device):
        super(MLP, self).__init__()
        self.device = device
        self.layers = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.batch_size = 32
        # self.max_grad_norm = 1

    def forward(self, x):
        x = self.layers(x)
        return x


model = MLP(device)
loss_fn = nn.CrossEntropyLoss()

num_epochs = 200
initial_epsilon =
final_epsilon = 0.02
batch_size = 32
batches_per_train = 1
batches_per_eval = 10
last_eval_batch_i = 0
batch_i = 0
replay_buffer = []
min_buffer_size = 50
max_buffer_size = 100

for epsilon in np.linspace(initial_epsilon, final_epsilon, num_epochs):
    # gather data
    trajectories = generate_trajectories(epsilon, num_episodes=1)
    replay_buffer.extend(trajectories)
    if len(replay_buffer) > max_buffer_size:
        replay_buffer = replay_buffer[len(replay_buffer)-max_buffer_size:]
    step_dataset = StepDataset(replay_buffer, model.device)
    data_loader = DataLoader(step_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f'#trajectories: {len(replay_buffer)}, #steps: {len(step_dataset)}, epsilon: {epsilon}')
    # train
    if len(step_dataset) < min_buffer_size:
        continue
    batches_in_dataset = len(step_dataset) // model.batch_size
    batches_to_train = min(batches_in_dataset, batches_per_train)
    batch_i = train_on_buffer(model, data_loader, batches_to_train, writer, batch_i)
    # eval
    if batch_i - last_eval_batch_i >= batches_per_eval:
        last_eval_batch_i = batch_i
        run_eval(model)
