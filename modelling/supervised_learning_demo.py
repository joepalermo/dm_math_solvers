import numpy as np
import math
import torch
from modelling.transformer_encoder import TransformerEncoderModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import read_text_file
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


def load_data_from_corpus(filepath):
    examples = read_text_file(filepath).split('\n')
    examples = [e for e in examples if e.count('gcd') < 2 and e.count('; Value') < 1]
    return examples


def input_to_target(x):
    if 'gcd' in x and 'param_0' in x and 'param_1' in x:
        y = 1
    elif 'gcd' in x and 'param_1' in x:
        y = 2
    elif 'gcd' in x:
        y = 3
    else:
        y = 0
    return y


def encode(raw_observation, tokenizer):
    encoded_ids = tokenizer.encode(raw_observation).ids
    # pad the encoded ids up to a maximum length
    encoded_ids.extend([padding_token for _ in range(seq_len - len(encoded_ids))])
    return np.array(encoded_ids)


def decode(ids, tokenizer):
    """call with: decode(encoded.ids, tokenizer)"""
    return "".join([tokenizer.id_to_token(id_) for id_ in ids if id_ != padding_token])


# data params
vocab_size = 200
padding_token = vocab_size
seq_len = 100

# architecture params
ntoken = vocab_size + 1
nhead = 4
nhid = 128
nlayers = 1
num_outputs = 4
dropout = 0

# training params
n_epochs = 100
batch_size = 64
lr = 1
lr_decay_factor = 0.8
max_grad_norm = 0.5

# prep dataset ---------------------------------------------------------------------------------------------------------

raw_xs = load_data_from_corpus('environment/corpus/gcd_corpus.txt')
ys = [input_to_target(x) for x in raw_xs]
tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(vocab_size=vocab_size)
tokenizer.train(trainer, ['environment/corpus/10k_corpus.txt'])
tokenizer.save('tokenizer.json')
xs = np.concatenate([np.expand_dims(encode(x, tokenizer), 0) for x in raw_xs], axis=0)
ys = np.array(ys)
num_examples = len(xs)

# # inspect data
# for i in range(100):
#     print(raw_xs[i])
#     print(xs[i])
#     print(decode(xs[i], tokenizer), 'answer:', ys[i])
#     print()

train_xs, valid_xs, train_ys, valid_ys = train_test_split(xs, ys, test_size=int(num_examples * 0.1))
train_xs = torch.from_numpy(train_xs)
train_ys = torch.from_numpy(train_ys)
valid_xs = torch.from_numpy(valid_xs)
valid_ys = torch.from_numpy(valid_ys)

# train and validate model ---------------------------------------------------------------------------------------------

model = TransformerEncoderModel(ntoken=ntoken, nhead=nhead, nhid=nhid, nlayers=nlayers, num_outputs=num_outputs,
                                dropout=dropout)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=lr_decay_factor)

for epoch in range(n_epochs):
    # shuffle training set indices
    permutation = torch.randperm(train_xs.size()[0])
    for batch_i, i in enumerate(list(range(0, train_xs.size()[0], batch_size))):
        batch_indices = permutation[i:i+batch_size]
        batch_xs, batch_ys = train_xs[batch_indices], train_ys[batch_indices]
        batch_logits = model(batch_xs)
        loss = criterion(batch_logits, batch_ys)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if i % 10 == 0:
            print(f'train_loss @ step #{batch_i}', loss.item())
            valid_logits = model(valid_xs)
            valid_preds = torch.argmax(valid_logits, axis=1)
            valid_preds = valid_preds.detach().numpy()
            valid_targets = valid_ys.detach().numpy()
            valid_acc = accuracy_score(valid_targets, valid_preds)
            print(f'valid_acc', valid_acc)
            if valid_acc == 1:
                torch.save(model, 'modelling/models/gcd.pt')
                break
    scheduler.step()

