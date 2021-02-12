import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader
from modelling.differentiate_supervised_analogue.utils import Dataset, generate_dataset
from modelling.differentiate_supervised_analogue.models import RNN, Transformer
import random

torch.manual_seed(42)
np.random.seed(seed=42)
device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

# define dataset ------------------------------

num_cases = 5
num_safe_distractors = 5
num_kill_distractors = 5
num_distractors = num_safe_distractors + num_kill_distractors
samples = generate_dataset(num_unique_samples=10000,
                           num_cases=num_cases,
                           num_safe_distractors=num_safe_distractors,
                           num_kill_distractors=num_kill_distractors,
                           sequence_length=15)

# unpack
train_samples, val_samples = train_test_split(samples, test_size=0.5)
train_case_inputs, train_sequence_inputs, train_targets = zip(*train_samples)
val_case_inputs, val_sequence_inputs, val_targets = zip(*val_samples)

# convert to torch
train_case_inputs = torch.from_numpy(np.array(train_case_inputs))
train_sequence_inputs = torch.from_numpy(np.array(train_sequence_inputs))
train_targets = torch.from_numpy(np.array(train_targets))
val_case_inputs = torch.from_numpy(np.array(val_case_inputs))
val_sequence_inputs = torch.from_numpy(np.array(val_sequence_inputs))
val_targets = torch.from_numpy(np.array(val_targets))
# print(train_case_inputs.shape, train_sequence_inputs.shape, train_targets.shape)
# print(val_case_inputs.shape, val_sequence_inputs.shape, val_targets.shape)

# build data loaders
train_dataset = Dataset(train_case_inputs, train_sequence_inputs, train_targets)
val_dataset = Dataset(val_case_inputs, val_sequence_inputs, val_targets)
train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
val_data_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, drop_last=True)

# define models ------------------------------

rnn_config = {'num_cases': num_cases,
          'action_padding_token': num_distractors + 2,
          'num_action_tokens': num_distractors + 3,
          'num_layers': 2,
          'hidden_size': 128,
          'dropout': 0.1,
          'lr': 0.001,
          'weight_decay': 0}

transformer_config = {'num_cases': num_cases,
          'action_padding_token': num_distractors + 2,
          'num_action_tokens': num_distractors + 3,
          'num_layers': 1,
          'hidden_size': 128,
          'dropout': 0.1,
          'lr': 0.0001,
          'weight_decay': 0}

model = RNN(rnn_config, device)
# model = Transformer(transformer_config, device)

# learn via SL ------------------------------
# criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.MSELoss()
total_batches = 0

# training loop
for epoch in range(5):
    for batch in train_data_loader:
        # unpack batch
        case_inputs = batch[0].to(model.device)
        sequence_inputs = batch[1].to(model.device)
        targets = batch[2].to(model.device).type(torch.float32)
        # train
        output = model(case_inputs, sequence_inputs)
        loss = criterion(output, targets)
        model.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        model.optimizer.step()
        total_batches += 1
        if total_batches % 10 == 0:
            print(f'train_loss @ step #{total_batches}', loss.item())
            # validation
            val_scores = list()
            for batch in val_data_loader:
                # unpack batch
                case_inputs = batch[0].to(model.device)
                sequence_inputs = batch[1].to(model.device)
                targets = batch[2].to(model.device)
                # validation
                val_output = model(case_inputs, sequence_inputs)
                val_output = val_output.detach().cpu().numpy()
                val_targets = targets.detach().cpu().numpy()
                val_score = mean_absolute_error(val_targets, val_output)
                val_scores.append(val_score)
            print(f'val MAE: ', np.array(val_scores).mean())

# visualize final val performance
for batch in val_data_loader:
    # unpack batch
    case_inputs = batch[0].to(model.device)
    sequence_inputs = batch[1].to(model.device)
    targets = batch[2].to(model.device)
    # validation
    val_output = model(case_inputs, sequence_inputs)
    val_output = val_output.detach().cpu().numpy()
    val_targets = targets.detach().cpu().numpy()
    val_score = mean_absolute_error(val_targets, val_output)
    val_scores.append(val_score)
    case_inputs, sequence_inputs = \
        case_inputs.detach().cpu().numpy(), sequence_inputs.detach().cpu().numpy()
    for case_input, sequence_input, output, target in zip(case_inputs, sequence_inputs, val_output, val_targets):
        print(case_input, sequence_input, output, target)