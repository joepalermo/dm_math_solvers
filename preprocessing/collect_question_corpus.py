from random import shuffle
from utils import read_text_file
from sklearn.model_selection import train_test_split

filenames = read_text_file("environment/module_lists/most_natural_composed_for_program_synthesis.txt").split("\n")
filepaths = [
    f"mathematics_dataset-v1.0/train-easy/{filename}" for filename in filenames
]

num_problems = 250000
num_problems_per_module = num_problems // len(filepaths)
questions = []

for filepath in filepaths:
    with open(filepath, "r") as f:
        lines = f.readlines()
    num_pairs = min(len(lines) // 2, num_problems_per_module)
    for i in range(0, 2 * num_pairs, 2):
        question = lines[i].strip()
        answer = lines[i + 1].strip()
        questions.append(question)

shuffle(questions)
train_questions, val_questions = train_test_split(questions, test_size=0.4)
with open("environment/tokenization/train_question_corpus.txt", "w") as f:
    f.write("\n".join(train_questions))
with open("environment/tokenization/val_question_corpus.txt", "w") as f:
    f.write("\n".join(val_questions))
