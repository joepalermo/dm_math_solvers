from hparams import HParams
hparams = HParams('.', hparams_filename='hparams', name='rl_math', ask_before_deletion=False)
import sentencepiece as spm
import matplotlib.pyplot as plt
from utils import read_text_file

# train tokenizer on question corpus
hardcoded_symbols = hparams.env.operators + ["'p_0'", "'p_1'"]
spm.SentencePieceTrainer.train(input='environment/tokenization/question_corpus.txt',
                               model_prefix='environment/tokenizer',
                               vocab_size=200,
                               user_defined_symbols=hardcoded_symbols)

# load tokenizer
sp = spm.SentencePieceProcessor(model_file='environment/tokenization/tokenizer.model')
example_state = "What is the third derivative of -t**4 - 880*t**3 + 152*t**2 wrt t?; df('p_0')"
id_tokens = sp.encode(example_state)
string_tokens = sp.encode(example_state, out_type=str)
assert sp.decode(id_tokens) == example_state
assert sp.decode(string_tokens) == example_state

# visualize encoded lengths using trained tokenizer on graph corpus
validation_corpus = read_text_file('environment/tokenization/graph_corpus.txt')
val_state = validation_corpus.split('\n')
val_length_dist = [len(sp.encode(vs)) for vs in val_state]
plt.hist(val_length_dist, density=True, bins=100)  # density=False would make counts
plt.show()

