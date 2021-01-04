from utils import read_pickle

trajectories = read_pickle('mathematics_dataset-v1.0/trajectories/trajectories.pkl')

# extract trajectories
all_trajectories = list()
for module_name, difficulty in trajectories.keys():
    selected_trajectories = trajectories[(module_name,difficulty)]
    all_trajectories.extend(selected_trajectories)

# create training and val sets
states, actions, rewards = [], [], []
for state, action, reward, _, _ in all_trajectories:
    states.append(state)
    actions.append(action)
    rewards.append(reward)
    print(state.shape, action, reward)

# load or instantiate model
# TODO

# training loop
# TODO

