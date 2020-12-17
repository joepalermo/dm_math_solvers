from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from utils import write_pickle, read_pickle


def decode(tokenizer, ids):
    return "".join([tokenizer.id_to_token(id_) for id_ in ids])

# tokenizer.pre_tokenizer = Whitespace()
# trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(vocab_size=280)
tokenizer.train(trainer, ["preprocessing/corpus/corpus.txt"])
# save_to_filepath = 'preprocessing/tokenizer'
# tokenizer.save(save_to_filepath)
# tokenizer = Tokenizer.from_file(save_to_filepath)
inp = "Let s = -34/19 - -4715/304. Find the common denominator of -55/4 and s.; make_function(diff(lookup_value(lookup_value('param_0','param_1'),extract_isolated_variable(make_equality(simplify('param_0'),'param_1')))),Expression('-55/4'))"
encoded = tokenizer.encode(inp)
decoded = decode(tokenizer, encoded.ids)
print(encoded.tokens)
print(encoded.ids)
print(decoded)




