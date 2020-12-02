import unittest
from environment import extract_formal_elements
from environment.operators.solve_linsys import extract_coefficients, solve_linsys
from environment.operators import append, lookup_value

class Test(unittest.TestCase):

    def test1(self):
        problem_statement = 'Solve $f[0 = 4*b + b + 15] for $f[b].'
        fs = extract_formal_elements(problem_statement)
        # print(fs)
        print(lookup_value(solve_linsys(fs[0]), fs[1]))

    # def test2(self):
    #     problem_statement = 'Solve $f[0 = 4*f - 0*t - 4*t - 4], $f[-4*f + t = -13] for $f[f].'
    #     fs = extract_formal_elements(problem_statement)
    #     # print(fs)
    #     print(lookup_value(solve_linsys(append(append(None, fs[0]), fs[1])), fs[1]))