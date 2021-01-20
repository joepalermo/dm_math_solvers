from random import sample, shuffle
from utils import read_text_file
from tqdm import tqdm

filenames = read_text_file("environment/module_lists/most_natural_composed_for_program_synthesis.txt").split("\n")
filepaths = [
    f"mathematics_dataset-v1.0/train-easy/{filename}" for filename in filenames
]

num_problems = 50000
num_problems_per_module = num_problems // len(filepaths)
p_val = 0.2
questions = []

for filepath in filepaths:
    with open(filepath, "r") as f:
        lines = f.readlines()
    num_pairs = min(len(lines) // 2, num_problems_per_module)
    for i in range(0, 2 * num_pairs, 2):
        question = lines[i].strip()
        answer = lines[i + 1].strip()
        questions.append(question + ';')

shuffle(questions)
val_questions = questions[:int(len(questions)*p_val)]
train_questions = questions[int(len(questions)*p_val):]
train_questions = list(set(train_questions)-set(val_questions))
assert len(set(val_questions).intersection(set(train_questions))) == 0
print(f'# training questions: {len(train_questions)}')
print(f'# validation questions: {len(val_questions)}')
with open("environment/tokenization/question_corpus_train.txt", "w") as f:
    f.write("\n".join(train_questions))
with open("environment/tokenization/question_corpus_val.txt", "w") as f:
    f.write("\n".join(val_questions))