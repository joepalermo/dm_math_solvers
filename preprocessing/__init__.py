import numpy as np
from utils import write_pickle, read_pickle, write_json
from tqdm import tqdm
import glob
import tensorflow as tf


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
    write_json(f'{tokenizers_dirpath}/idx2char.json', idx2char)

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


def get_paired_filepaths(filepaths):
    '''return list of tuples that are (question,answer) pairs'''
    identifier_dict = dict()
    for fp in filepaths:
        identifier = "_".join(fp.split('/')[-1].split('_')[:-1])
        if identifier not in identifier_dict:
            identifier_dict[identifier] = [fp]
        else:
            identifier_dict[identifier].append(fp)
            identifier_dict[identifier] = tuple(sorted(identifier_dict[identifier], reverse=True))
    return list(identifier_dict.values())


def load_train(mode, num_files_to_include=None):
    if mode == 'easy':
        train_file_pattern = 'output/train_easy_preprocessed/*.npy'
    elif mode == 'all':
        train_file_pattern = 'output/train*/*.npy'
    train_filepaths = glob.glob(train_file_pattern)
    paired_filepaths = get_paired_filepaths(train_filepaths)
    if num_files_to_include is not None:
        paired_filepaths = paired_filepaths[:num_files_to_include]
    all_q = list()
    all_a = list()
    for q_filepath, a_filepath in tqdm(paired_filepaths):
        q = np.load(q_filepath)
        a = np.load(a_filepath)
        all_q.append(q)
        all_a.append(a)
    q = np.concatenate(all_q, axis=0)
    a = np.concatenate(all_a, axis=0)
    return q, a


def build_train_and_val_datasets(q_train, a_train, params, val_split=0.1, buffer_size=20000, batch_size=64):
    np.random.shuffle(q_train)
    np.random.shuffle(a_train)
    dataset = tf.data.Dataset.from_tensor_slices((q_train, a_train))
    input_data = dataset.take(params.num_examples).shuffle(q_train.shape[0]).batch(params.batch_size) \
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    train_ds = input_data.take(params.num_training_batches).repeat(params.num_epochs)
    val_ds = input_data.skip(params.num_training_batches)
    return train_ds, val_ds