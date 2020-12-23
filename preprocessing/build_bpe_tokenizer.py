from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from utils import write_pickle, read_pickle


def decode(tokenizer, ids):
    """call with: decode(tokenizer, encoded.ids)"""
    return "".join([tokenizer.id_to_token(id_) for id_ in ids])


# tokenizer.pre_tokenizer = Whitespace()
# trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(vocab_size=280, special_tokens=["[PAD]"])
tokenizer.train(trainer, ["environment/corpus/10k_corpus.txt"])
save_to_filepath = "preprocessing/tokenizer"
tokenizer.save(save_to_filepath)
# tokenizer = Tokenizer.from_file(save_to_filepath)
with open("environment/corpus/1k_corpus.txt") as f:
    corpus = f.read()
observations = corpus.split("\n")
max_observation_length = max([len(tokenizer.encode(obs)) for obs in observations])
print(max_observation_length)
