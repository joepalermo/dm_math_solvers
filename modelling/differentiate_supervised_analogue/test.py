import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from torch.utils.data import DataLoader
from modelling.differentiate_supervised_analogue.utils import Dataset
from modelling.differentiate_supervised_analogue.models import RNN, Transformer
import random
torch.manual_seed(42)
np.random.seed(seed=42)
device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')


def generate_dataset(num_samples, num_cases, num_distractors, sequence_length):
    '''
    samples can have target 0 if they have either:
    1) the wrong number of 0s to match the case
    2) distractor symbols
    otherwise
    '''
    samples = []
    start_token = num_distractors + 1
    padding_token = num_distractors + 2
    for _ in range(num_samples):
        sequence = []
        case = random.randint(0, num_cases-1)
        target = random.randint(0, 1)
        # target = 1
        if target == 1:
            sequence.extend([0] * case)
        elif random.random() < 0.5:
            wrong_num_cases = case
            while wrong_num_cases == case:
                wrong_num_cases = random.randint(0, num_cases-1)
            sequence.extend([0] * wrong_num_cases)
        else:
            sequence.extend([0] * case)
            sampled_num_distractors = random.randint(0, sequence_length - len(sequence) - 1)
            distractors = [random.randint(1, num_distractors) for _ in range(sampled_num_distractors)]
            sequence.extend(distractors)
        random.shuffle(sequence)
        # put sequence together
        padding = [padding_token for _ in range(sequence_length - len(sequence))]
        sequence = [start_token] + sequence + padding
        sample = (case, sequence, target)
        samples.append(sample)
    return samples


num_cases = 3
num_distractors = 3
samples = generate_dataset(num_samples=50000,
                           num_cases=num_cases,
                           num_distractors=num_distractors,
                           sequence_length=8)

for sample in samples:
    print(sample)

# unpack
train_samples, val_samples = train_test_split(samples)
train_case_inputs, train_sequence_inputs, train_targets = zip(*train_samples)
val_case_inputs, val_sequence_inputs, val_targets = zip(*val_samples)

# convert to torch
train_case_inputs = torch.from_numpy(np.array(train_case_inputs))
train_sequence_inputs = torch.from_numpy(np.array(train_sequence_inputs))
train_targets = torch.from_numpy(np.array(train_targets))
val_case_inputs = torch.from_numpy(np.array(val_case_inputs))
val_sequence_inputs = torch.from_numpy(np.array(val_sequence_inputs))
val_targets = torch.from_numpy(np.array(val_targets))

train_dataset = Dataset(train_case_inputs, train_sequence_inputs, train_targets)
val_dataset = Dataset(val_case_inputs, val_sequence_inputs, val_targets)
train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
val_data_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=True)

config = {'num_cases': num_cases,
          'action_padding_token': num_distractors + 2,
          'num_action_tokens': num_distractors + 3,
          'num_layers': 2,
          'hidden_size': 128,
          'dropout': 0.1,
          'lr': 0.001,
          'weight_decay': 0.001}

model = RNN(config, device)
model = Transformer(config, device)

# learn via SL ---------------------------------------------------------------------------------------------
# criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.MSELoss()
total_batches = 0
for epoch in range(1):
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
            val_accs = list()
            for batch in val_data_loader:
                # unpack batch
                case_inputs = batch[0].to(model.device)
                sequence_inputs = batch[1].to(model.device)
                targets = batch[2].to(model.device)

                val_output = model(case_inputs, sequence_inputs)
                val_output = val_output.detach().cpu().numpy()
                val_targets = targets.detach().cpu().numpy()
                val_acc = mean_squared_error(val_targets, val_output)
                val_accs.append(val_acc)
            print(f'val_acc', np.array(val_accs).mean())