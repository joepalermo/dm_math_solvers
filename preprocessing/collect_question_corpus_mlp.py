#Derivatives only for now
from hparams import HParams
hparams = HParams('.', hparams_filename='hparams', name='rl_math', ask_before_deletion=False)
from random import shuffle
from utils import read_text_file
from environment.utils import tokenize_formal_elements

filenames = read_text_file("environment/module_lists/most_natural_composed_for_program_synthesis.txt").split("\n")
filepaths = [
    f"mathematics_dataset-v1.0/train-easy/{filename}" for filename in filenames
]

num_problems = 50000
num_problems_per_module = num_problems // len(filepaths)
questions = []

for filepath in filepaths:
    with open(filepath, "r") as f:
        lines = f.readlines()
    num_pairs = min(len(lines) // 2, num_problems_per_module)
    for i in range(0, 2 * num_pairs, 2):
        question = lines[i].strip()
        question = tokenize_formal_elements(question)
        answer = lines[i + 1].strip()
        questions.append(question + ';')

shuffle(questions)
with open("environment/tokenization/question_corpus_mlp.txt", "w") as f:
    f.write("\n".join(questions))
