import numpy as np
from utils import write_pickle, read_pickle, write_json, recreate_dirpath

def build_tokenizer(input_filepaths, tokenizers_dirpath):
    print(f"building vocab from {len(input_filepaths)} files")
    # read all files into one large string
    list_of_texts = []
    for fp in input_filepaths:
        print(f"reading {fp}...")
        with open(fp, 'r') as f:
            list_of_texts.append(f.read())
    all_text = ''.join(list_of_texts)

    # get the unique characters in the file (vocab)
    chars_to_exclude = {'\n'}
    print("computing vocab...")
    vocab = list(set(all_text) - chars_to_exclude)

    # initialize vocab with pad (_), start (@), and stop (~) tokens
    assert '_' not in vocab
    assert '@' not in vocab
    assert '~' not in vocab
    special_char2idx = {'_': 0, '@': 1, '~': 2}

    # creating a mapping from all unique characters to indices
    char2idx = {
                **special_char2idx,
                **{v: i+len(special_char2idx) for i, v in enumerate(vocab)}
               }
    idx2char = {i: v for v, i in char2idx.items()}

    # save pprint
    write_json(f'{tokenizers_dirpath}/char2idx.json', char2idx)

    # save tokenizer
    write_pickle(f'{tokenizers_dirpath}/char2idx.pkl', char2idx)
    write_pickle(f'{tokenizers_dirpath}/idx2char.pkl', idx2char)
    print("done building tokenizer")


def load_tokenizers(tokenizers_dirpath):
    char2idx = read_pickle(f'{tokenizers_dirpath}/char2idx.pkl')
    idx2char = read_pickle(f'{tokenizers_dirpath}/idx2char.pkl')
    return char2idx, idx2char


def preprocess_data(input_filepaths, char2idx, output_dirpath):
    print(f"preprocessing data from {len(input_filepaths)} files")
    for fp in input_filepaths:
        fn = fp.split('/')[-1]
        print(f"preprocessing {fn}...")
        questions_encoded = []
        answers_encoded = []
        with open(fp, 'r') as f:
            lines = f.readlines()
        num_pairs = len(lines) // 2
        print(f"\tencoding characters...")
        for i in range(0, 2 * num_pairs, 2):
            question = lines[i].strip()
            answer = lines[i+1].strip()
            questions_encoded.append([char2idx[ch] for ch in question])
            answers_encoded.append([char2idx[a] for a in answer])
        print(f"\tpadding...")
        # TODO: perform padding in numpy for acceleration
        questions_padded = np.array([q + [0] * (160 - len(q)) for q in questions_encoded])
        answers_padded = np.array([[1] + a + [2] + [0] * (30 - len(a)) for a in answers_encoded])
        print(f"\twriting to file...")
        # write preprocessed data to file
        identifier = fn.split('.')[0]
        np.save(f'{output_dirpath}/{identifier}_questions.npy', questions_padded)
        np.save(f'{output_dirpath}/{identifier}_answers.npy', answers_padded)


def get_max_question_and_answer_length(input_filepaths):
    max_question_length = 0
    max_answer_length = 0
    for fp in input_filepaths:
        fn = fp.split('/')[-1]
        print(f"preprocessing {fn}...")
        with open(fp, 'r') as f:
            lines = f.readlines()
        num_pairs = len(lines) // 2
        for i in range(0, 2 * num_pairs, 2):
            question_length = len(lines[i].strip())
            answer_length = len(lines[i+1].strip())
            if question_length > max_question_length:
                max_question_length = question_length
            if answer_length > max_answer_length:
                max_answer_length = answer_length
        print(max_question_length, max_answer_length)
    return max_question_length, max_answer_length