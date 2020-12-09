import glob
from preprocessing import extract_formal_elements
from utils import write_json


def extract_unit_tests(questions):
    annotated_questions = {}
    for question in questions:
        formal_elements = extract_formal_elements(question)
        annotated_questions[question] = formal_elements
    write_json('preprocessing/unit_testing/extract_formal_elements_examples.json', annotated_questions)


def extract_questions(all_filepaths, num_files, questions_per_file):
    questions = []
    questions_per_file = 3
    num_files = 100
    for filepath in all_filepaths[:num_files]:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        num_pairs = min(len(lines) // 2, questions_per_file)
        for i in range(0, 2 * num_pairs, 2):
            question = lines[i].strip()
            answer = lines[i + 1].strip()
            questions.append(question)
    return questions


# def create_annotated_data(all_filepaths, destination_dirpath):
#     pairs = []
#     for filepath in all_filepaths:
#         with open(filepath, 'r') as f:
#             lines = f.readlines()
#         num_pairs = len(lines) // 2
#         for i in range(0, 2 * num_pairs, 2):
#             question = lines[i].strip()
#             answer = lines[i + 1].strip()
#             annotated_question
#             pairs.append((question, answer))
#     return pairs


train_easy_file_pattern = 'mathematics_dataset-v1.0/train-easy/*.txt'
train_easy_filepaths = glob.glob(train_easy_file_pattern)
train_easy_composed_filepaths = [fp for fp in train_easy_filepaths if 'compose' in fp]
all_filepaths = train_easy_composed_filepaths


# Errors from train-easy:
# TODO: Fix: Simplify (0 + (sqrt(1008) + sqrt(1008) + 1)*-4)**2. ['(0 + (', '(1008) +', '(1008) + 1)*-4)**2']
# TODO: Fix: What is 3 (base 11) in base 5? ['3 (', '11)', '5']
# TODO: Fix: How many micrometers are there in twenty-one quarters of a millimeter? ['-', 'a']
# TODO: don't filter out tpppbbpbbb from: What is prob of picking 1 b and 1 p when two letters picked without replacement from tpppbbpbbb?
# TODO: What is prob of sequence ccbc when four letters picked without replacement from nnscspb? []
# TODO: What is 481 minutes after 7:26 PM? ['481', '7:26']
# TODO: What is the k'th term of 485, 472, 459, 446? ["k'", '485', '472', '459', '446']


