import os

dirpath = 'train-easy'
filepaths = [os.path.join(dirpath, fn) for fn in os.listdir(dirpath)]

first_questions = list()
for fp in filepaths:
    with open(fp) as f:
        text = f.read()
    first_problem = text.split('\n')[:10]
    q1,a1,q2,a2,q3,a3,q4,a4,q5,a5 = first_problem
    fn = fp.split('/')[-1].split('.txt')[0]
    if "composed" not in fn:
        first_questions.append(f'{fn}: \n{a1}\n{a2}\n{a3}\n{a4}\n{a5}')

first_questions = sorted(first_questions)
with open('notes/first_answers.txt', 'w') as f:
    text = '\n\n'.join(first_questions)
    f.write(text)
