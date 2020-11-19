import os
import shutil
import json
import pickle


def recreate_dirpath(dirpath):
    if os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
    os.mkdir(dirpath)


def read_text_file(filepath):
    with open(filepath) as f:
        return f.read()


def write_text_file(filepath, text):
    with open(filepath, 'w') as f:
        f.write(text)


def read_json(filepath):
    if not os.path.isfile(filepath):
        return {}
    with open(filepath, 'r') as f:
        json_string = f.read()
    return json.loads(json_string)


def write_json(filepath, dict_to_write):
    json_string = json.dumps(dict_to_write, indent=4)
    with open(filepath, 'w') as f:
        f.write(json_string)


def read_pickle(filepath):
    return pickle.load(open(filepath, "rb"))


def write_pickle(filepath, obj):
    pickle.dump(obj, open(filepath, "wb"))
