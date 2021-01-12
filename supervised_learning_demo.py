from hparams import HParams
hparams = HParams('.', hparams_filename='hparams', name='rl_math', ask_before_deletion=False)

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
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(42)
np.random.seed(seed=42)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(0)


def load_data_from_corpus(filepath):
    examples = read_text_file(filepath).split('\n')
    examples = [e for e in examples if 'None' not in e]  # filter out multi-variate problems
    examples = [e for e in examples if ('first' in e) or ('second' in e) or ('third' in e)]  # filter out multi-variate problems
    questions = [e.split(';')[0] for e in examples]
    questions_with_graphs = [sample_graph(q) for q in questions]
    questions_with_targets = [(q,input_to_target(q)) for q in questions_with_graphs]
    return questions_with_targets


def input_to_target(x):
    '''let y=0 mean df,
       let y=1 mean Eq(...)'''
    if 'third' in x and ' df(df(df(p_' in x:
        y = 1
    elif 'second' in x and ' df(df(p_' in x:
        y = 1
    elif 'first' in x and ' df(p_' in x:
        y = 1
    else:
        y = 0
    return y

def sample_graph(x):
    import random
    df_0 = ''
    df_1 = ' df(p_'
    df_2 = ' df(df(p_'
    df_3 = ' df(df(df(p_'
    if 'third' in x:
        g = random.sample([df_0, df_1, df_2, df_3], k=1)[0]
    elif 'second' in x:
        g = random.sample([df_0, df_1, df_2], k=1)[0]
    elif 'first' in x:
        g = random.sample([df_0, df_1], k=1)[0]
    return f'{x}; {g}'


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
n_epochs = 8
batch_size = 64
lr = 1
lr_decay_factor = 0.8
max_grad_norm = 0.5

# prep dataset ---------------------------------------------------------------------------------------------------------

data = load_data_from_corpus('environment/corpus/validation_graphs_calculus__differentiate_0.txt')
print(len(data))
data = data[:1000]
raw_xs = [d[0] for d in data]
ys = [d[1] for d in data]
print('data loaded')

padding_token = vocab_size
special_tokens = ['df', 'p_']
tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
tokenizer.train(trainer, ['environment/corpus/20k_question_corpus.txt'])
print('tokenizer created')

xs = np.concatenate([np.expand_dims(encode(x, tokenizer), 0) for x in raw_xs], axis=0)
ys = np.array(ys)
num_examples = len(xs)

# inspect data
for i in range(100):
    print('raw question: ', raw_xs[i])
    print('encoded question: ', xs[i])
    print('decoded question: ', decode(xs[i], tokenizer))
    print('target: ', ys[i])
    print()
#
train_xs, valid_xs, train_ys, valid_ys = train_test_split(xs, ys, test_size=int(num_examples * 0.1))
train_xs = torch.from_numpy(train_xs)
train_ys = torch.from_numpy(train_ys)
valid_xs = torch.from_numpy(valid_xs)
valid_ys = torch.from_numpy(valid_ys)

# train and validate model ---------------------------------------------------------------------------------------------

model = TransformerEncoderModel(ntoken=ntoken, nhead=nhead, nhid=nhid, nlayers=nlayers, num_outputs=num_outputs,
                                dropout=dropout, device=device).cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=lr_decay_factor)
writer = SummaryWriter(comment="_seed_42_yes_padding")

total_batches = 0
for epoch in range(n_epochs):
    # shuffle training set indices
    permutation = torch.randperm(train_xs.size()[0])
    for batch_i, i in enumerate(list(range(0, train_xs.size()[0], batch_size))):
        batch_indices = permutation[i:i+batch_size]
        batch_xs, batch_ys = train_xs[batch_indices], train_ys[batch_indices]
        batch_logits = model(batch_xs.cuda())
        loss = criterion(batch_logits, batch_ys.cuda())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_batches += 1
        if total_batches % 10 == 0:
            print(f'train_loss @ step #{total_batches}', loss.item())
            valid_logits = model(valid_xs.cuda())
            valid_preds = torch.argmax(valid_logits, axis=1)
            valid_preds = valid_preds.detach().cpu().numpy()
            valid_targets = valid_ys.detach().cpu().numpy()
            valid_acc = accuracy_score(valid_targets, valid_preds)
            print(f'valid_acc', valid_acc)
            writer.add_scalar('Val/acc', valid_acc, total_batches)
            writer.add_scalar('Train/loss', loss.item(), total_batches)
            writer.close()
    scheduler.step()

