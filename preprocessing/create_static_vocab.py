from hparams import HParams
hparams = HParams('.', hparams_filename='hparams', name='rl_math', ask_before_deletion=False)
import sentencepiece as spm

hardcoded_symbols = hparams.env.operators + ["'p_0'", "'p_1'"]
spm.SentencePieceTrainer.train(input='environment/corpus/20k_corpus.txt',
                               model_prefix='preprocessing/tokenizer',
                               vocab_size=200,
                               user_defined_symbols=hardcoded_symbols)


sp = spm.SentencePieceProcessor(model_file='environment/tokenizer.model')
print(sp.encode('What is the third derivative of -t**4 - 880*t**3 + 152*t**2 wrt t?', out_type=str))
print(sp.encode("What is the third derivative of -t**4 - 880*t**3 + 152*t**2 wrt t?; df('p_0')", out_type=str))
encoding = sp.encode("What is the third derivative of -t**4 - 880*t**3 + 152*t**2 wrt t?; df('p_0')", out_type=str)
print(sp.decode(encoding))
