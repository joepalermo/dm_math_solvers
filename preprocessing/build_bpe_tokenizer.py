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

special_tokens = [
            "lv",
            "ss",
            "ap",
            "ape",
            "meq",
            "lve",
            "eiv",
            "slr",
            "fac",
            "df",
            "dfw",
            "sy",
            "mfn",
            "ra",
            "mod",
            "gcd",
            "md0",
            "ip",
            "lcm",
            "pf",
            "fa",
            "nt",
            "'p_0'",
            "'p_1'"
        ]

tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(vocab_size=280, special_tokens=special_tokens)
tokenizer.train(trainer, ["environment/corpus/20k_question_corpus.txt"])

save_to_filepath = "preprocessing/tokenizer"
tokenizer.save(save_to_filepath)
# tokenizer = Tokenizer.from_file(save_to_filepath)
