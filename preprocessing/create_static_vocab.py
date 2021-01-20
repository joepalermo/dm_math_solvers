from hparams import HParams
hparams = HParams('.', hparams_filename='hparams', name='rl_math', ask_before_deletion=False)
import sentencepiece as spm
import matplotlib.pyplot as plt
from utils import read_text_file

# hardcoded_symbols = hparams.env.operators + ["'p_0'", "'p_1'"]
# spm.SentencePieceTrainer.train(input='environment/tokenization/question_corpus_train.txt',
#                                model_prefix='environment/tokenizer',
#                                vocab_size=200,
#                                user_defined_symbols=hardcoded_symbols)


sp = spm.SentencePieceProcessor(model_file='environment/tokenization/tokenizer.model')
# print(sp.encode('What is the third derivative of -t**4 - 880*t**3 + 152*t**2 wrt t?', out_type=str))
# print(sp.encode("What is the third derivative of -t**4 - 880*t**3 + 152*t**2 wrt t?; df('p_0')", out_type=str))
encoding = sp.encode("What is the third derivative of -t**4 - 880*t**3 + 152*t**2 wrt t?; df('p_0')")
print(encoding)
print(sp.decode(encoding))

# validation_corpus = read_text_file('environment/tokenization/question_corpus_val.txt')
# val_questions = validation_corpus.split('\n')
# val_length_dist = [len(sp.encode(q)) for q in val_questions]
# plt.hist(val_length_dist, density=True, bins=100)  # density=False would make counts
# plt.show()

