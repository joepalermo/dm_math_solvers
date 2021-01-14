from hparams import HParams
hparams = HParams('.', hparams_filename='hparams', name='rl_math', ask_before_deletion=False)

import numpy as np
import torch
from modelling.transformer_encoder import TransformerEncoderModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from demo_utils import load_data_from_corpus

torch.manual_seed(42)
np.random.seed(seed=42)

device = torch.device(f'cuda:{hparams.run.gpu_id}' if torch.cuda.is_available() else 'cpu')

def encode(raw_observation, tokenizer):
    encoded_ids = tokenizer.encode(raw_observation).ids
    # pad the encoded ids up to a maximum length
    encoded_ids.extend([padding_token for _ in range(hparams.env.max_sequence_length - len(encoded_ids))])
    return np.array(encoded_ids)


def decode(ids, tokenizer):
    """call with: decode(encoded.ids, tokenizer)"""
    return "".join([tokenizer.id_to_token(id_) for id_ in ids if id_ != padding_token])


# prep dataset ---------------------------------------------------------------------------------------------------------

num_examples = 10000
data = load_data_from_corpus('environment/corpus/validation_graphs_calculus__differentiate_0.txt')
data = data[:num_examples]
raw_xs = [d[0] for d in data]
ys = [d[1] for d in data]

padding_token = hparams.env.vocab_size
special_tokens = ['df', 'p_']
tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(vocab_size=hparams.env.vocab_size, special_tokens=special_tokens)
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

# split train and val
train_xs, valid_xs, train_ys, valid_ys = train_test_split(xs, ys, test_size=int(num_examples * 0.1))

# setup SL dataset ---------------------------------------------------------------------------------------------

# convert to torch
train_xs = torch.from_numpy(train_xs)
train_ys = torch.from_numpy(train_ys)
valid_xs = torch.from_numpy(valid_xs)
valid_ys = torch.from_numpy(valid_ys)

# train and validate model ---------------------------------------------------------------------------------------------

n_outputs = 2
model = TransformerEncoderModel(ntoken=hparams.env.vocab_size + 1,
                                nhead=hparams.model.nhead,
                                nhid=hparams.model.nhid,
                                nlayers=hparams.model.nlayers,
                                num_outputs=n_outputs,
                                dropout=hparams.model.dropout,
                                device=device,
                                lr=hparams.train.lr,
                                max_grad_norm=hparams.train.max_grad_norm,
                                batch_size=hparams.train.batch_size)

# learn via SL ---------------------------------------------------------------------------------------------
criterion = torch.nn.CrossEntropyLoss()
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.model.max_grad_norm)
        model.optimizer.step()
        total_batches += 1
        if total_batches % 10 == 0:
            print(f'train_loss @ step #{total_batches}', loss.item())
            valid_logits = model(valid_xs.to(model.device))
            valid_preds = torch.argmax(valid_logits, axis=1)
            valid_preds = valid_preds.detach().cpu().numpy()
            valid_targets = valid_ys.detach().cpu().numpy()
            valid_acc = accuracy_score(valid_targets, valid_preds)
            print(f'valid_acc', valid_acc)
            # writer.add_scalar('Val/acc', valid_acc, total_batches)
            # writer.add_scalar('Train/loss', loss.item(), total_batches)
            # writer.close()
    model.scheduler.step()

