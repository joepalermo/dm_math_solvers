# from hparams import HParams
# hparams = HParams('.', hparams_filename='hparams', name='rl_math', ask_before_deletion=False)
import sentencepiece as spm
import matplotlib.pyplot as plt
from utils import read_text_file

# train tokenizer on question corpus
# hardcoded_symbols = hparams.env.operators + hparams.env.types + ["'p_0'", "'p_1'", "'", 'G']  # why is 'G' needed?
hardcoded_symbols = ['G']  # why is 'G' needed?
spm.SentencePieceTrainer.train(input='environment/tokenization/train_question_corpus.txt',
                               model_prefix='environment/tokenization/tokenizer',
                               question_vocab_size=250,
                               user_defined_symbols=hardcoded_symbols)


# load tokenizer
sp = spm.SentencePieceProcessor(model_file='environment/tokenization/tokenizer.model')
example_state = "What is the third derivative of -t**4 - 880*t**3 + 152*t**2 wrt t?"
id_tokens = sp.encode(example_state)
string_tokens = sp.encode(example_state, out_type=str)
assert sp.decode(id_tokens) == example_state, sp.decode(id_tokens)
assert sp.decode(string_tokens) == example_state, sp.decode(string_tokens)

# visualize encoded lengths using trained tokenizer on graph corpus
graph_corpus = read_text_file('environment/tokenization/val_question_corpus.txt')
raw_observations = graph_corpus.split('\n')
for raw_obs in raw_observations:
    assert raw_obs == sp.decode(sp.encode(raw_obs))
val_length_dist = [len(sp.encode(raw_obs)) for raw_obs in raw_observations]
print('max length: ', max(val_length_dist))
plt.hist(val_length_dist, density=True, bins=100)  # density=False would make counts
plt.show()

