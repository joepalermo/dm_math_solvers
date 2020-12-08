import re
import glob

train_easy_file_pattern = 'mathematics_dataset-v1.0/train-easy/*.txt'
train_easy_filepaths = glob.glob(train_easy_file_pattern)
all_filepaths = train_easy_filepaths

questions = []
questions_per_file = 1
num_files = 100
for filepath in all_filepaths[:num_files]:
    with open(filepath, 'r') as f:
        lines = f.readlines()
    num_pairs = min(len(lines) // 2, questions_per_file)
    for i in range(0, 2 * num_pairs, 2):
        question = lines[i].strip()
        answer = lines[i + 1].strip()
        questions.append(question)

annotated_questions = []
for question in questions:
    print(question)
    split_on_punctuation = "***".join([string for string in re.split('[?.,;]', question)
                                       if len(string) > 0 and not string.isspace()])
    split_on_words = [string for string in re.split('[A-Za-z]\w+|\*\*\*', split_on_punctuation)
                      if len(string) > 0 and not string.isspace()]
    formal_elements = [string.strip() for string in split_on_words]
    print(formal_elements)


# TODO: Fix: Is 15 a factor of 720? ['15 a', '720']
# TODO: Let o = -788/3 - -260. Which is bigger: -0.1 or o? ['o = -788/3 - -260', ': -0', '1', 'o']
# TODO: Fix: Simplify (0 + (sqrt(1008) + sqrt(1008) + 1)*-4)**2. ['(0 + (', '(1008) +', '(1008) + 1)*-4)**2']
# TODO: Fix: What is 3 (base 11) in base 5? ['3 (', '11)', '5']
# TODO: Fix: How many micrometers are there in twenty-one quarters of a millimeter? ['-', 'a']
# TODO: don't filter out tpppbbpbbb from: What is prob of picking 1 b and 1 p when two letters picked without replacement from tpppbbpbbb?
# TODO: What is prob of sequence ccbc when four letters picked without replacement from nnscspb? []
# TODO: What is 481 minutes after 7:26 PM? ['481', '7:26']
# TODO: What is -0.0006832 rounded to 5 decimal places? ['-0', '0006832', '5']
# TODO: Work out 4 * 4.45. ['4 * 4', '45']
# TODO: What is the k'th term of 485, 472, 459, 446? ["k'", '485', '472', '459', '446']


