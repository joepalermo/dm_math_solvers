import os

easy_dirpath = "train-easy"
medium_dirpath = "train-medium"
hard_dirpath = "train-hard"
easy_filepaths = [os.path.join(easy_dirpath, fn) for fn in os.listdir(easy_dirpath)]
medium_filepaths = [
    os.path.join(medium_dirpath, fn) for fn in os.listdir(medium_dirpath)
]
hard_filepaths = [os.path.join(hard_dirpath, fn) for fn in os.listdir(hard_dirpath)]
easy_composed_filepaths = [
    os.path.join(easy_dirpath, fn) for fn in os.listdir(easy_dirpath) if "compose" in fn
]
medium_composed_filepaths = [
    os.path.join(medium_dirpath, fn)
    for fn in os.listdir(medium_dirpath)
    if "compose" in fn
]
hard_composed_filepaths = [
    os.path.join(hard_dirpath, fn) for fn in os.listdir(hard_dirpath) if "compose" in fn
]


def write_first_questions(filepaths, destination_filepath):
    first_questions = list()
    for fp in filepaths:
        with open(fp) as f:
            text = f.read()
        first_problem = text.split("\n")[:10]
        q1, a1, q2, a2, q3, a3, q4, a4, q5, a5 = first_problem
        fn = fp.split("/")[-1].split(".txt")[0]
        first_questions.append(
            f"{fp}: \n{q1}\n\t{a1}\n{q2}\n\t{a2}\n{q3}\n\t{a3}\n{q4}\n\t{a4}\n{q5}\n\t{a5}"
        )
    first_questions = sorted(first_questions)
    with open(destination_filepath, "a") as f:
        text = "\n\n".join(first_questions)
        f.write(text)


def print_module_names(filepaths):
    for fp in filepaths:
        fn = fp.split("/")[-1]
        print(fn)


# destination_filepath = 'notes/samples_from_all_modules.txt'
# with open(destination_filepath, 'w') as f:
#     f.write('')
# write_first_questions(easy_filepaths, destination_filepath)
# write_first_questions(medium_filepaths, destination_filepath)
# write_first_questions(hard_filepaths, destination_filepath)

destination_filepath = "notes/samples_from_all_composed_modules.txt"
with open(destination_filepath, "w") as f:
    f.write("")
write_first_questions(easy_composed_filepaths, destination_filepath)
write_first_questions(medium_composed_filepaths, destination_filepath)
write_first_questions(hard_composed_filepaths, destination_filepath)

print_module_names(sorted(easy_filepaths))
print_module_names(sorted(easy_composed_filepaths))
