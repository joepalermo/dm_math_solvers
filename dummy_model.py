import random
import numpy as np
import math
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
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
dropout = 0.2
slice_encoding = True

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


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerEncoderModel(torch.nn.Module):
    def __init__(self, ntoken, nhead, nhid, nlayers, dropout):
        super().__init__()
        torch.nn.Module.__init__(self)
        self.token_embedding = torch.nn.Embedding(ntoken, nhid)
        self.pos_encoder = PositionalEncoding(nhid, dropout)
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=nhid, nhead=nhead), nlayers
        )
        # if slicing a single element of the encoded sequence
        if slice_encoding:
            self.policy_output = torch.nn.Linear(nhid, 4)
        else:
            self.policy_output = torch.nn.Linear(nhid*seq_len, 4)
        # self.value_output = torch.nn.Linear(nhid, 1)

    def forward(self, token_idxs):
        # embed the tokens
        embedding = self.token_embedding(token_idxs).permute((1, 0, 2))
        embedding_with_pos = self.pos_encoder(embedding)
        # create the padding mask
        # padding_mask = torch.where(token_idxs == padding_token, 0, 1).type(torch.BoolTensor)
        # nn.transformer requires shape (seq_len, batch_size, embedding_dim)
        # apply the transformer encoder
        encoding = self.transformer_encoder(embedding_with_pos)  # , src_key_padding_mask=padding_mask)

        # use sliced encoding
        if slice_encoding:
            sliced_encoding = encoding[0]
            logits = self.policy_output(sliced_encoding)
        # use flattened encoding
        else:
            encoding = encoding.permute((1, 0, 2))
            flattened_encoding = torch.flatten(encoding, start_dim=1)
            logits = self.policy_output(flattened_encoding)

        # squeeze because values are scalars, not 1D array
        # logit = self.value_output(sliced_encoding).squeeze(-1)
        # print(embedding)
        # print(embedding_with_pos)
        # print(padding_mask)
        # print(encoding)
        # print(sliced_encoding)
        # print(logit)
        return logits


# Construct our model by instantiating the class defined above
model = TransformerEncoderModel(ntoken=ntoken, nhead=nhead, nhid=nhid, nlayers=nlayers, dropout=dropout)
# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the nn.Linear
# module which is members of the model.
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=lr_decay_factor)

for epoch in range(n_epochs):
    permutation = torch.randperm(train_xs.size()[0])
    for batch_i, i in enumerate(list(range(0, train_xs.size()[0], batch_size))):
        optimizer.zero_grad()
        indices = permutation[i:i+batch_size]
        batch_xs, batch_ys = train_xs[indices], train_ys[indices]
        batch_preds = model(batch_xs)
        # batch_ys = batch_ys.type_as(batch_preds)
        loss = criterion(batch_preds, batch_ys)
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if i % 10 == 0:
            print(f'train_loss @ step #{batch_i}', loss.item())
            valid_preds = model(valid_xs)
            valid_preds = torch.argmax(valid_preds, axis=1)
            print('mean valid pred:', valid_preds.type(torch.float).mean())
            print('mean valid target:', valid_ys.type(torch.float).mean())
            valid_targets = valid_ys.detach().numpy()
            valid_preds = valid_preds.detach().numpy()
            valid_acc = accuracy_score(valid_targets, valid_preds)
            print(f'valid_acc', valid_acc)
    scheduler.step()

