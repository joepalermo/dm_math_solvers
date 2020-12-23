import glob
from preprocessing import get_max_question_and_answer_length

# define inputs
input_file_pattern = "mathematics_dataset-v1.0/train*/*.txt"
input_filepaths = glob.glob(input_file_pattern)

# eda
max_question_length, max_answer_length = get_max_question_and_answer_length(
    input_filepaths
)
print(max_question_length, max_answer_length)
