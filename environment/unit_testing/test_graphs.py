import unittest
from environment.utils import extract_formal_elements
from environment.operators import append, add_keypair, lookup_value, function_application, apply_mapping, calc, \
    make_equality, equality_lhs, simplify, solve_system


# [op1, op2, ... f1, f2, ...]

class Test(unittest.TestCase):

    @staticmethod
    def test_easy_algebra__linear_1d():
        problem_statement = 'Solve $f[0 = 4*b + b + 15] for $f[b].'
        f = extract_formal_elements(problem_statement)
        assert lookup_value(solve_system(f[0]), f[1]) == -3

    def test_easy_algebra__linear_1d_composed(self):
        problem_statement = 'Let $f[w] be $f[(-1 + 13)*3/(-6)]. Let $f[b = w - -6]. Let $f[i = 2 - b]. Solve $f[-15 = 3*c + i*c] for $f[c].'
        f = extract_formal_elements(problem_statement)
        eq = make_equality(f[0], calc(f[1]))
        b = simplify(f[2])
        system = append(append(append(None, eq), b), f[3])
        soln = solve_system(system)
        i = lookup_value(soln, equality_lhs(f[3]))
        i_mapping = add_keypair(None, equality_lhs(f[3]), i)
        lin_eq = apply_mapping(f[4], i_mapping)
        assert lookup_value(solve_system(lin_eq), f[5]) == -3

    def test_easy_algebra__linear_2d(self):
        problem_statement = 'Solve $f[0 = 4*f - 0*t - 4*t - 4], $f[-4*f + t = -13] for $f[f].'
        f = extract_formal_elements(problem_statement)
        assert lookup_value(solve_system(append(append(None, f[0]), f[1])), f[2]) == 4

    def test_train_easy_algebra__linear_2d_composed(self):
        problem_statement = 'Suppose $f[2*y + 12 = 6*y]. Suppose $f[y = f - 15]. Solve $f[-8 = -4*w], $f[-3*d - 4*w + f = -8*d] for $f[d].'
        f = extract_formal_elements(problem_statement)
        system = append(append(append(append(None, f[0]), f[1]), f[2]), f[3])
        assert lookup_value(solve_system(system), f[4]) == -2

    def test_train_easy_algebra__polynomial_roots(self):
        problem_statement = 'Solve $f[-3*h**2/2 - 24*h - 45/2 = 0] for $f[h].'
        f = extract_formal_elements(problem_statement)
        assert lookup_value(solve_system(f[0]), f[1]) == {-1, -15}


    # medium ---------
    def test_medium_algebra__linear_1d_compose(self):
        problem_statement = 'Let $f[n(m) = m**3 - 7*m**2 + 13*m - 2]. Let $f[j] be $f[n(4)]. Solve $f[0 = 3*x + j*x + 10] for $f[x].'
        f = extract_formal_elements(problem_statement)
        assert lookup_value(solve_system(apply_mapping(f[3], add_keypair(None, f[1], function_application(f[0], f[2])))), f[4]) == -2


