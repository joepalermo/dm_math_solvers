import unittest
from environment.utils import extract_formal_elements
from environment.operators import append, add_keypair, lookup_value, function_application, apply_mapping, calc, \
    make_equality, project_lhs, project_rhs, simplify, solve_system, factor, diff, replace_arg, substitution_left_to_right, \
    eval_in_base, root, round_to_int, round_to_dec, power, substitution_right_to_left, max_arg, min_arg, greater_than, \
    less_than, lookup_value_eq


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
        eq1 = make_equality(f[0], f[1])
        system = append(append(append(None, eq1), f[2]), f[3])
        soln = solve_system(system)
        i_eq = lookup_value_eq(soln, project_lhs(f[3]))
        lin_eq = substitution_left_to_right(f[4], i_eq)
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

    def test_train_easy_algebra__polynomial_roots_1(self):
        problem_statement = 'Solve $f[-3*h**2/2 - 24*h - 45/2 = 0] for $f[h].'
        f = extract_formal_elements(problem_statement)
        assert lookup_value(solve_system(f[0]), f[1]) == {-1, -15}

    def test_train_easy_algebra__polynomial_roots_2(self):
        problem_statement = 'Factor $f[-n**2/3 - 25*n - 536/3].'
        f = extract_formal_elements(problem_statement)
        assert factor(f[0]) == '-(n + 8)*(n + 67)/3'

    def test_train_easy_algebra__polynomial_roots_3(self):
        problem_statement = 'Find $f[s] such that $f[9*s**4 - 8958*s**3 - 14952*s**2 - 2994*s + 2991 = 0].'
        f = extract_formal_elements(problem_statement)
        assert lookup_value(solve_system(f[1]), f[0]) == {-1, 1/3, 997}

    def test_train_easy_algebra__polynomial_roots_composed_1(self):
        problem_statement = 'Let $f[d = -25019/90 - -278]. Let $f[v(j)] be the third derivative of $f[0 + 1/27*j**3 - d*j**5 + 1/54*j**4 + 3*j**2 + 0*j]. Suppose $f[v(o) = 0]. What is $f[o]?'
        f = extract_formal_elements(problem_statement)
        d = simplify(f[0])
        function = substitution_left_to_right(f[2], d)
        v = diff(diff(diff(function)))
        v_eq = make_equality(f[1], v)
        v_eq_o = replace_arg(v_eq, f[4])
        equation = substitution_left_to_right(f[3], v_eq_o)  # e.g. x.subs(sym.sympify('f(x)'), sym.sympify('v'))
        assert lookup_value(solve_system(equation), f[4]) == {-1/3, 1}

    def test_train_easy_arithmetic__add_or_sub_in_base(self):
        problem_statement = 'In base $f[13], what is $f[7a79 - -5]?'
        f = extract_formal_elements(problem_statement)
        assert eval_in_base(f[1], f[0]) == '7a81'

    def test_train_easy_arithmetic__nearest_integer_root_1(self):
        problem_statement = 'What is the $f[2] root of $f[664] to the nearest $f[1]?'
        f = extract_formal_elements(problem_statement)
        root_f1 = root(f[1], f[0])
        rounded_root_f1 = round_to_int(root_f1, f[2])
        assert rounded_root_f1 == '26'

    def test_train_easy_arithmetic__nearest_integer_root_2(self):
        problem_statement = 'What is $f[1699] to the power of $f[1/6], to the nearest $f[1]?'
        f = extract_formal_elements(problem_statement)
        power_f0 = power(f[0], f[1])
        rounded_power_f0 = round_to_int(power_f0, f[2])
        assert rounded_power_f0 == '3'

    # def test_train_easy_comparison__closest(self):
    #     problem_statement = 'Which is the closest to -1/3?  (a) -8/7  (b) 5  (c) -1.3'
    #     f = extract_formal_elements(problem_statement)
    #     power_f0 = power(f[0], f[1])
    #     rounded_power_f0 = round_to_int(power_f0, f[2])
    #     assert rounded_power_f0 == '3'


    def test_train_easy_comparison__pair_composed(self):
        problem_statement = 'Let $f[o = -788/3 - -260]. Which is bigger: $f[-0.1] or $f[o]?'
        f = extract_formal_elements(problem_statement)
        o = simplify(f[0])
        m = max_arg(f[1], project_rhs(o))
        assert substitution_right_to_left(m, o) == '-0.1'


    # def test_train_easy_comparison__sort_composed(self):
    #     problem_statement = 'Suppose $f[0 = -4*x + 8*x - 40]. Let $f[h(i) = i**2 - 9*i - 14]. Let $f[n] be $f[h(x)]. Sort $f[-1], $f[4], $f[n].'
    #     f = extract_formal_elements(problem_statement)
    #     x = lookup_value(solve_system(f[0]), get


    # medium ---------
    def test_medium_algebra__linear_1d_compose(self):
        problem_statement = 'Let $f[n(m) = m**3 - 7*m**2 + 13*m - 2]. Let $f[j] be $f[n(4)]. Solve $f[0 = 3*x + j*x + 10] for $f[x].'
        f = extract_formal_elements(problem_statement)
        assert lookup_value(solve_system(apply_mapping(f[3], add_keypair(None, f[1], function_application(f[0], f[2])))), f[4]) == -2




