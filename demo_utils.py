import random
from utils import read_text_file


def load_data_from_corpus(filepath):
    examples = read_text_file(filepath).split('\n')
    examples = [e for e in examples if 'None' not in e]  # filter out multi-variate problems
    examples = [e for e in examples if ('first' in e) or ('second' in e) or ('third' in e)]  # filter out multi-variate problems
    questions = [e.split(';')[0] for e in examples]
    questions_with_graphs = [sample_graph(q) for q in questions]
    questions_with_targets = [(q,input_to_target(q)) for q in questions_with_graphs]
    return questions_with_targets


def sample_graph(x):
    df_0 = ''
    df_1 = ' df(p_'
    df_2 = ' df(df(p_'
    df_3 = ' df(df(df(p_'
    if 'third' in x:
        g = random.sample([df_0, df_1, df_2, df_3], k=1)[0]
    elif 'second' in x:
        g = random.sample([df_0, df_1, df_2], k=1)[0]
    elif 'first' in x:
        g = random.sample([df_0, df_1], k=1)[0]
    return f'{x}; {g}'


def input_to_target(x):
    '''let y=0 mean df,
       let y=1 mean Eq(...)'''
    if 'third' in x and ' df(df(df(p_' in x:
        y = 1
    elif 'second' in x and ' df(df(p_' in x:
        y = 1
    elif 'first' in x and ' df(p_' in x:
        y = 1
    else:
        y = 0
    return y
