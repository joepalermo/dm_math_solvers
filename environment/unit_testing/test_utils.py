import unittest
from environment.utils import is_numeric, tokenize_formal_elements


class Test(unittest.TestCase):
    def test_is_numeric(self):
        assert is_numeric("2")
        assert is_numeric("2.0")
        assert not is_numeric("2.0.")

    def test_tokenize_formal_elements(self):
        question = "Find the third derivative of -a**3*g**3*t**3 + 642*a**3*g*t**3 + 16*a**3*g*t**2 - 5*a**2*t**2 + a*g**3 wrt t."
        # print(tokenize_formal_elements(question))
        assert tokenize_formal_elements(question) == "Find the third derivative of $Expression wrt $Variable ."