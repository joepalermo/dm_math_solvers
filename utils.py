import os
import sys
import shutil
import json
import pickle
import logging
from datetime import datetime as dt
from functools import reduce
from operator import mul


flatten = lambda l: [item for sublist in l for item in sublist]


def recreate_dirpath(dirpath):
    if os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
    os.mkdir(dirpath)


def read_text_file(filepath):
    with open(filepath) as f:
        return f.read()


def write_text_file(filepath, text):
    with open(filepath, "w") as f:
        f.write(text)


def read_json(filepath):
    if not os.path.isfile(filepath):
        return {}
    with open(filepath, "r") as f:
        json_string = f.read()
    return json.loads(json_string)


def write_json(filepath, dict_to_write):
    json_string = json.dumps(dict_to_write, indent=4)
    with open(filepath, "w") as f:
        f.write(json_string)


def read_pickle(filepath):
    return pickle.load(open(filepath, "rb"))


def write_pickle(filepath, obj):
    pickle.dump(obj, open(filepath, "wb"))


def get_logger(name, experiment_dirpath):

    logger_path = experiment_dirpath
    if not os.path.exists(logger_path):
        os.makedirs(logger_path)

    # add logging
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # # log all uncaught exceptions
    # def exception_handler(type, value, tb):
    #     logger.exception("Uncaught exception: {0}".format(str(value)))
    # # Install exception handler
    # sys.excepthook = exception_handler

    # https://stackoverflow.com/questions/6729268/log-messages-appearing-twice-with-python-logging
    if not logger.handlers:
        # create a file handler
        current_time = dt.now().strftime("%Y%m%d")
        file_handler = logging.FileHandler(
            os.path.join(logger_path, "{}_{}.log".format(current_time, name))
        )
        file_handler.setLevel(logging.INFO)
        # create a logging format
        formats = "[%(asctime)s - %(name)s-%(lineno)d - %(funcName)s - %(levelname)s] %(message)s"
        file_formatter = logging.Formatter(formats, "%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        # add the handlers to the logger
        logger.addHandler(file_handler)

        # console handler
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.INFO)
        c_formatter = logging.Formatter(formats, "%m-%d %H:%M:%S")
        c_handler.setFormatter(c_formatter)
        logger.addHandler(c_handler)
    return logger

