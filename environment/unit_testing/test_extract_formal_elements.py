import unittest
from utils import read_json, write_json
from environment.utils import extract_formal_elements


def run_test_case(question, formal_elements):
    extracted_formal_elements = extract_formal_elements(question)
    assert extracted_formal_elements == formal_elements


class Test(unittest.TestCase):
    def test_examples(self):
        question_to_formal_elements = read_json(
            "environment/unit_testing/artifacts/extract_formal_elements_examples.json"
        )
        for question, formal_elements in question_to_formal_elements.items():
            run_test_case(question, formal_elements)
