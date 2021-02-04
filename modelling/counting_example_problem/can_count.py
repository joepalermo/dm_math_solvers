from hparams import HParams
hparams = HParams('..', hparams_filename='hparams', name='rl_math', ask_before_deletion=False)
from utils import flatten, read_pickle
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from modelling.transformer_encoder import TransformerEncoderModel
from torch.utils.data import DataLoader

torch.manual_seed(42)
np.random.seed(seed=42)

device = torch.device(f'cuda:{hparams.run.gpu_id}' if torch.cuda.is_available() else 'cpu')


def get_vocab(examples):
    vocab = dict()
    inpts = [inpt for inpt, _ in examples]
    joined_inputs = "".join(inpts)
    unique_characters = set(joined_inputs)
    for i, ch in enumerate(unique_characters):
        vocab[ch] = i
    return vocab


def encode(inpt, vocab):
    padding_token = len(vocab)
    encoded_ids = [vocab[ch] for ch in inpt]
    # pad the encoded ids up to a maximum length
    encoded_ids.extend([padding_token for _ in range(max_sequence_length - len(encoded_ids))])
    return np.array(encoded_ids)


def decode(inpt, vocab):

    inv_vocab = {v: k for k, v in vocab.items()}
    padding_token = len(inv_vocab)
    decoded_inpt = [inv_vocab[num] for num in inpt if num != padding_token]
    return ''.join(decoded_inpt)


max_sequence_length = 100
examples = read_pickle('preprocessing/counting_dataset.pkl')
train_examples, val_examples = train_test_split(examples, test_size=0.2)
vocab = get_vocab(train_examples)
padding_token = len(vocab)
train_inpts, train_targets = zip(*train_examples)
val_inpts, val_targets = zip(*val_examples)
encoded_train_examples = [encode(inpt, vocab) for inpt in train_inpts]
encoded_val_examples = [encode(inpt, vocab) for inpt in val_inpts]

train_xs = np.concatenate([np.expand_dims(encoded_inpt, 0) for encoded_inpt in encoded_train_examples], axis=0)
val_xs = np.concatenate([np.expand_dims(encoded_inpt, 0) for encoded_inpt in encoded_val_examples], axis=0)
train_ys = np.array([int(y) for y in train_targets])
val_ys = np.array([int(y) for y in val_targets])

# inspect data
for i in range(1):
    print('raw question/answer: ', train_examples[i])
    print('encoded question: ', encoded_train_examples[i])
    print('decoded question: ', decode(train_xs[i], vocab))
    assert train_examples[i][0] == decode(train_xs[i], vocab)
    print('target: ', train_ys[i])
    print()

# setup SL dataset ---------------------------------------------------------------------------------------------

# convert to torch
train_xs = torch.from_numpy(train_xs)
train_ys = torch.from_numpy(train_ys)
val_xs = torch.from_numpy(val_xs)
val_ys = torch.from_numpy(val_ys)

# train and validate model ---------------------------------------------------------------------------------------------

n_outputs = 2
ntoken = len(vocab) + 1
num_outputs = 6
model = TransformerEncoderModel(ntoken=ntoken, num_outputs=num_outputs, device=device)

class Dataset(torch.utils.data.Dataset):

    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]

val_dataset = Dataset(val_xs, val_ys)
val_data_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=True)



# learn via SL ---------------------------------------------------------------------------------------------
criterion = torch.nn.CrossEntropyLoss()
# criterion = torch.nn.MSELoss()
total_batches = 0
for epoch in range(10):
    # shuffle training set indices
    permutation = torch.randperm(train_xs.size()[0])
    for batch_i, i in enumerate(list(range(0, train_xs.size()[0], hparams.train.batch_size))):
        batch_indices = permutation[i:i+hparams.train.batch_size]
        batch_xs, batch_ys = train_xs[batch_indices], train_ys[batch_indices]
        batch_logits = model(batch_xs.to(model.device))
        loss = criterion(batch_logits, batch_ys.to(model.device))
        model.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.train.max_grad_norm)
        model.optimizer.step()
        total_batches += 1
        if total_batches % 10 == 0:
            print(f'train_loss @ step #{total_batches}', loss.item())
            val_accs = list()
            for val_batch_xs, val_batch_ys in val_data_loader:
                valid_logits = model(val_batch_xs.to(model.device))
                valid_preds = torch.argmax(valid_logits, axis=1)
                valid_preds = valid_preds.detach().cpu().numpy()
                valid_targets = val_batch_ys.detach().cpu().numpy()
                valid_acc = accuracy_score(valid_targets, valid_preds)
                val_accs.append(valid_acc)
            print(f'valid_acc', np.array(val_accs).mean())
                # writer.add_scalar('Val/acc', valid_acc, total_batches)
                # writer.add_scalar('Train/loss', loss.item(), total_batches)
                # writer.close()


# 11100 => 3
# 10000 => 1
# 11110 => 4

