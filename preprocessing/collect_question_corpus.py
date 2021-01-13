from random import sample, shuffle
from utils import read_text_file
from tqdm import tqdm

filenames = read_text_file("environment/module_lists/most_natural_composed_for_program_synthesis.txt").split("\n")
filepaths = [
    f"mathematics_dataset-v1.0/train-easy/{filename}" for filename in filenames
]

num_problems = 20000
num_problems_per_module = 20000 // len(filepaths)
all_observations = []

for filepath in filepaths:
    with open(filepath, "r") as f:
        lines = f.readlines()
    num_pairs = min(len(lines) // 2, num_problems_per_module)
    for i in range(0, 2 * num_pairs, 2):
        question = lines[i].strip()
        answer = lines[i + 1].strip()
        all_observations.append(question + ';')

shuffle(all_observations)
all_observations = "\n".join(all_observations)
with open("environment/corpus/20k_question_corpus.txt", "w") as f:
    f.write(all_observations)