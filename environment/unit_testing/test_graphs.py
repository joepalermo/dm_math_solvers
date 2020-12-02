import unittest
from environment.utils import extract_formal_elements
from environment.operators.solve_linsys import solve_linsys
from environment.operators import append, add_keypair, lookup_value, function_application, apply_mapping, calc, \
    make_equality


class Test(unittest.TestCase):

    @staticmethod
    def test_easy_algebra__linear_1d():
        problem_statement = 'Solve $f[0 = 4*b + b + 15] for $f[b].'
        f = extract_formal_elements(problem_statement)
        assert lookup_value(solve_linsys(f[0]), f[1]) == -3

    def test_easy_algebra__linear_1d_composed(self):
        problem_statement = 'Let $f[w] be $f[(-1 + 13)*3/(-6)]. Let $f[b = w - -6]. Let $f[i = 2 - b]. Solve $f[-15 = 3*c + i*c] for $f[c].'
        f = extract_formal_elements(problem_statement)
        eq = make_equality(f[0], calc(f[1]))
        system = append(append(append(None, eq), 'b = w + 6'), f[3])
        soln = solve_linsys(system)
        i = lookup_value(soln, 'i')
        lin_eq = apply_mapping(f[4], {'i': i})
        assert lookup_value(solve_linsys(lin_eq), f[5]) == -3

    def test_easy_algebra__linear_2d(self):
        problem_statement = 'Solve $f[0 = 4*f - 0*t - 4*t - 4], $f[-4*f + t = -13] for $f[f].'
        f = extract_formal_elements(problem_statement)
        assert lookup_value(solve_linsys(append(append(None, f[0]), f[1])), f[2]) == 4

    def train_easy_algebra__linear_2d_composed(self):
        problem_statement = 'Suppose 2*y + 12 = 6*y. Suppose y = f - 15. Solve -8 = -4*w, -3*d - 4*w + f = -8*d for d.'
        f = extract_formal_elements(problem_statement)

    # medium ---------
    def test_medium_algebra__linear_1d_compose(self):
        problem_statement = 'Let $f[n(m) = m**3 - 7*m**2 + 13*m - 2]. Let $f[j] be $f[n(4)]. Solve $f[0 = 3*x + j*x + 10] for $f[x].'
        f = extract_formal_elements(problem_statement)
        assert lookup_value(solve_linsys(apply_mapping(f[3], add_keypair(None, f[1], function_application(f[0], f[2])))), f[4]) == -2
