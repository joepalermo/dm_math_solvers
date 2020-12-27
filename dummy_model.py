import random
import numpy as np
import math
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def inpt_to_boolean(ls):
    count_0 = ls.count(0)
    count_1 = ls.count(1)
    return 0 if count_0 > count_1 else 1


num_examples = 5000
min_length = 10
max_length = 20

# first element copy problem
xs = [[random.choice([0,1]) for _ in range(1)] for _ in range(num_examples)]
# counting problem
# xs = [[random.choice([0,1]) for _ in range(random.randint(min_length, max_length))] for _ in range(num_examples)]
# padding
xs = [xs_ + [2 for _ in range(max_length-len(xs_))] for xs_ in xs]
# filter to extract unique input strings
# unique_xs = set([str(xs_)[1:-1] for xs_ in xs])
# print(len(unique_xs))
# xs = [xs_.split(',') for xs_ in unique_xs]
# xs = [[int(x) for x in xs_] for xs_ in xs]
# get targets
ys = [inpt_to_boolean(x) for x in xs]
xs, ys = np.array(xs), np.array(ys)
train_xs, valid_xs, train_ys, valid_ys = train_test_split(xs, ys, test_size=int(num_examples * 0.1))
# for x,y in zip(xs, ys):
#     print(x,y)
# print(train_xs.shape, train_ys.shape, valid_xs.shape, valid_ys.shape)

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
    def __init__(self, ntoken, nhid, nhead, nlayers, dropout):
        super().__init__()
        torch.nn.Module.__init__(self)
        self.token_embedding = torch.nn.Embedding(ntoken, nhid)
        self.pos_encoder = PositionalEncoding(nhid, dropout)
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=nhid, nhead=nhead), nlayers
        )
        # if slicing a single element of the encoded sequence
        # self.policy_output = torch.nn.Linear(nhid, 3)
        self.value_output = torch.nn.Linear(nhid, 1)

    def forward(self, token_idxs):
        # embed the tokens
        embedding = self.token_embedding(token_idxs)
        embedding_with_pos = self.pos_encoder(embedding)
        # create the padding mask
        padding_mask = torch.where(token_idxs == 2, 1, 0).type(torch.BoolTensor)
        # nn.transformer requires shape (seq_len, batch_size, embedding_dim)
        embedding_with_pos = embedding_with_pos.permute((1, 0, 2))
        # apply the transformer encoder
        encoding = self.transformer_encoder(embedding_with_pos, src_key_padding_mask=padding_mask)
        # produce outputs
        sliced_encoding = encoding[0]
        # flattened_encoding = torch.flatten(encoding, start_dim=1)
        # logits = self.policy_output(sliced_encoding)
        # squeeze because values are scalars, not 1D array
        logit = self.value_output(sliced_encoding).squeeze(-1)
        # print(embedding)
        # print(embedding_with_pos)
        # print(padding_mask)
        # print(encoding)
        # print(sliced_encoding)
        # print(logit)
        return logit


# Construct our model by instantiating the class defined above
model = TransformerEncoderModel(ntoken=3, nhid=32, nhead=2, nlayers=1, dropout=0.1)


n_epochs = 10
batch_size = 128



# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the nn.Linear
# module which is members of the model.
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
for epoch in range(n_epochs):
    permutation = torch.randperm(train_xs.size()[0])
    for batch_i, i in enumerate(list(range(0, train_xs.size()[0], batch_size))):
        optimizer.zero_grad()
        indices = permutation[i:i+batch_size]
        batch_xs, batch_ys = train_xs[indices], train_ys[indices]
        batch_preds = model(batch_xs)
        batch_ys = batch_ys.type_as(batch_preds)
        loss = criterion(batch_preds, batch_ys)
        if i % 10 == 0:
            print(f'train_loss @ step #{batch_i}', loss.item())
            batch_preds = model(batch_xs)
            batch_preds = torch.nn.Sigmoid()(batch_preds)
            batch_preds = torch.round(batch_preds)
            targets = batch_ys.detach().numpy()
            preds = batch_preds.detach().numpy()
            # print(targets, preds)
            valid_acc = accuracy_score(targets, preds)
            print(f'valid_acc', valid_acc)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

