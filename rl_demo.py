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
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(42)
np.random.seed(seed=42)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(0)


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

# setup RL dataset ---------------------------------------------------------------------------------------------

# create: states, actions, rewards
# inputs => states, targets => actions, rewards
train_states = train_xs
train_actions = train_ys
train_rewards = np.ones(len(ys))
# flip some actions and set corresponding rewards to negative
train_flip = np.random.randint(0, 2, size=len(train_actions))  # get array of boolean flags (indices to flip)
train_actions = np.array([1-action if flip else action for action, flip in zip(train_actions, train_flip)])
train_rewards = np.array([-reward if flip else reward for reward, flip in zip(train_rewards, train_flip)])

# convert to torch
train_states = torch.from_numpy(train_states)
train_actions = torch.from_numpy(train_actions)
train_rewards = torch.from_numpy(train_rewards)
valid_xs = torch.from_numpy(valid_xs)
valid_ys = torch.from_numpy(valid_ys)

# assemble training buffer
train_buffer = [(s,a,r) for s,a,r in zip(train_states, train_actions, train_rewards)]

# train and validate model ---------------------------------------------------------------------------------------------

n_outputs = 2
model = TransformerEncoderModel(ntoken=hparams.model.vocab_size + 1,
                                nhead=hparams.model.nhead,
                                nhid=hparams.model.nhid,
                                nlayers=hparams.model.nlayers,
                                num_outputs=n_outputs,
                                dropout=hparams.model.dropout,
                                device=device,
                                lr=hparams.train.lr,
                                max_grad_norm=hparams.train.max_grad_norm,
                                batch_size=hparams.train.batch_size).cuda()

# learn via RL ---------------------------------------------------------------------------------------------
from modelling.train_utils import train_on_buffer, get_logdir
logdir = get_logdir()
writer = SummaryWriter(log_dir=logdir)
for i in range(10):
    # train -----------------
    train_on_buffer(model, train_buffer, writer, 0, hparams.train.batches_per_train)
    # validate -----------------
    valid_logits = model(valid_xs.cuda())
    valid_preds = torch.argmax(valid_logits, axis=1)
    print(f'some preds:', valid_preds[:20])
    valid_preds = valid_preds.detach().cpu().numpy()
    valid_targets = valid_ys.detach().cpu().numpy()
    valid_acc = accuracy_score(valid_targets, valid_preds)
    print(f'valid_acc:', valid_acc)




