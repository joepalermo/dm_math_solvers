import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from modelling.train_utils import train_on_buffer, StepDataset, get_logdir
device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(log_dir=get_logdir())


def generate_trajectories(num_episodes=20, episode_timeout=100):
    trajectories = list()
    env = gym.make('CartPole-v0')
    for i_episode in range(num_episodes):
        state = env.reset()
        trajectory = []
        for t in range(episode_timeout):
            env.render()
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            trajectory.append((state, action, reward, next_state, done))
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                trajectories.append(trajectory)
                break
    env.close()
    return trajectories


class MLP(nn.Module):
    def __init__(self, device):
        super(MLP, self).__init__()
        self.device = device
        self.layers = nn.Sequential(
            nn.Linear(784, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        # x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x


model = MLP(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()


batch_size = 32
batches_per_train = 10

for _ in range(1):
    # gather data
    trajectories = generate_trajectories()
    step_dataset = StepDataset(trajectories, model.device)
    data_loader = DataLoader(step_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    # train
    batches_in_dataset = len(step_dataset) // model.batch_size
    batches_to_train = min(batches_in_dataset, batches_per_train)
    batch_i = train_on_buffer(model, data_loader, batches_to_train, writer, batch_i)
    # eval