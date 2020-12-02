import unittest
from environment import extract_formal_elements
from environment.operators.solve_linsys import solve_linsys
from environment.operators import append, add_keypair, lookup_value, function_application, apply_mapping

class Test(unittest.TestCase):

    def test_algebra__linear_1d(self):
        problem_statement = 'Solve $f[0 = 4*b + b + 15] for $f[b].'
        fs = extract_formal_elements(problem_statement)
        assert lookup_value(solve_linsys(fs[0]), fs[1]) == -3

    def test_algebra__linear_2d(self):
        problem_statement = 'Solve $f[0 = 4*f - 0*t - 4*t - 4], $f[-4*f + t = -13] for $f[f].'
        fs = extract_formal_elements(problem_statement)
        assert lookup_value(solve_linsys(append(append(None, fs[0]), fs[1])), fs[2]) == 4

    def test_algebra__linear_1d_composed(self):
        problem_statement = 'Let $f[n(m) = m**3 - 7*m**2 + 13*m - 2]. Let $f[j] be $f[n(4)]. Solve $f[0 = 3*x + j*x + 10] for $f[x].'
        fs = extract_formal_elements(problem_statement)
        assert lookup_value(solve_linsys(apply_mapping(fs[3], add_keypair(None, fs[1], function_application(fs[0], fs[2])))), fs[4]) == -2
