import unittest
from environment.utils import extract_formal_elements, cast_formal_element
from environment.typed_operators import lookup_value, solve_system, append, make_equality, lookup_value_eq, project_lhs, \
    substitution_left_to_right, extract_isolated_variable
from environment.typed_operators import Equation, Function, Expression, Variable, Value


class Test(unittest.TestCase):

    def test_easy_algebra__linear_1d(self):
        problem_statement = 'Solve 0 = 4*b + b + 15 for b.'
        fs = extract_formal_elements(problem_statement)
        assert fs == [Equation('0 = 4*b + b + 15'), Variable('b')]
        system = append([], fs[0])
        solution = solve_system(system)
        value = lookup_value(solution, fs[1])
        assert value == Value(-3)

    def test_easy_algebra__linear_1d_composed(self):
        problem_statement = 'Let w be (-1 + 13)*3/(-6). Let b = w - -6. Let i = 2 - b. Solve -15 = 3*c + i*c for c.'
        f = extract_formal_elements(problem_statement)
        assert f == [Variable('w'), Expression('(-1 + 13)*3/(-6)'), Equation('b = w - -6'), Equation('i = 2 - b'),
                     Equation('-15 = 3*c + i*c'), Variable('c')]
        eq1 = make_equality(f[0], f[1])
        system = append(append(append([], eq1), f[2]), f[3])
        soln = solve_system(system)
        i_eq = lookup_value_eq(soln, extract_isolated_variable(f[3]))
        lin_eq = substitution_left_to_right(f[4], i_eq)
        assert lookup_value(solve_system(append([], lin_eq)), f[5]) == Value(-3)

    def test_easy_algebra__linear_2d(self):
        problem_statement = 'Solve 0 = 4*f - 0*t - 4*t - 4, -4*f + t = -13 for f.'
        f = extract_formal_elements(problem_statement)
        assert f == [Equation('0 = 4*f - 0*t - 4*t - 4'), Equation('-4*f + t = -13'), Variable('f')]

        assert lookup_value(solve_system(append(append([], f[0]), f[1])), f[2]) == 4

    def test_train_easy_algebra__linear_2d_composed(self):
        problem_statement = 'Suppose 2*y + 12 = 6*y. Suppose y = f - 15. Solve -8 = -4*w, -3*d - 4*w + f = -8*d for d.'
        f = extract_formal_elements(problem_statement)
        assert f == ['2*y + 12 = 6*y', 'y = f - 15', '-8 = -4*w', '-3*d - 4*w + f = -8*d', 'd']
        system = append(append(append(append(None, f[0]), f[1]), f[2]), f[3])
        assert lookup_value(solve_system(system), f[4]) == -2

    def test_train_easy_algebra__polynomial_roots_1(self):
        problem_statement = 'Solve -3*h**2/2 - 24*h - 45/2 = 0 for h.'
        f = extract_formal_elements(problem_statement)
        assert f == ['-3*h**2/2 - 24*h - 45/2 = 0', 'h']
        assert lookup_value(solve_system(f[0]), f[1]) == {-1, -15}

    def test_train_easy_algebra__polynomial_roots_2(self):
        problem_statement = 'Factor -n**2/3 - 25*n - 536/3.'
        f = extract_formal_elements(problem_statement)
        assert f == ['-n**2/3 - 25*n - 536/3']
        assert factor(f[0]) == '-(n + 8)*(n + 67)/3'

    def test_train_easy_algebra__polynomial_roots_3(self):
        problem_statement = 'Find s such that 9*s**4 - 8958*s**3 - 14952*s**2 - 2994*s + 2991 = 0.'
        f = extract_formal_elements(problem_statement)
        assert f == ['s', '9*s**4 - 8958*s**3 - 14952*s**2 - 2994*s + 2991 = 0']
        assert lookup_value(solve_system(f[1]), f[0]) == {-1, 1/3, 997}

    def test_train_easy_algebra__polynomial_roots_composed_1(self):
        problem_statement = 'Let d = -25019/90 - -278. Let v(j) be the third derivative of 0 + 1/27*j**3 - d*j**5 + 1/54*j**4 + 3*j**2 + 0*j. Suppose v(o) = 0. What is o?'
        f = extract_formal_elements(problem_statement)
        assert f == ['d = -25019/90 - -278', 'v(j)', '0 + 1/27*j**3 - d*j**5 + 1/54*j**4 + 3*j**2 + 0*j', 'v(o) = 0', 'o']
        d = simplify(f[0])
        function = substitution_left_to_right(f[2], d)
        v = diff(diff(diff(function)))
        v_eq = make_equality(f[1], v)
        v_eq_o = replace_arg(v_eq, f[4])
        equation = substitution_left_to_right(f[3], v_eq_o)  # e.g. x.subs(sym.sympify('f(x)'), sym.sympify('v'))
        assert lookup_value(solve_system(equation), f[4]) == {-1/3, 1}

    # def test_train_easy_comparison__closest(self):
    #     problem_statement = 'Which is the closest to -1/3?  (a) -8/7  (b) 5  (c) -1.3'
    #     f = extract_formal_elements(problem_statement)
    #     power_f0 = power(f[0], f[1])
    #     rounded_power_f0 = round_to_int(power_f0, f[2])
    #     assert rounded_power_f0 == '3'

    def test_train_easy_comparison__pair_composed(self):
        problem_statement = 'Let o = -788/3 - -260. Which is bigger: -0.1 or o?'
        f = extract_formal_elements(problem_statement)
        assert f == ['o = -788/3 - -260', '-0.1', 'o']
        o = simplify(f[0])
        m = max_arg(f[1], project_rhs(o))
        assert substitution_right_to_left(m, o) == '-0.1'

    # def test_train_easy_comparison__sort_composed(self):
    #     problem_statement = 'Suppose $f[0 = -4*x + 8*x - 40]. Let $f[h(i) = i**2 - 9*i - 14]. Let $f[n] be $f[h(x)]. Sort $f[-1], $f[4], $f[n].'
    #     f = extract_formal_elements(problem_statement)
    #     x = lookup_value(solve_system(f[0]), get

    # other -----------

    # def test_train_easy_arithmetic__add_or_sub_in_base(self):
    #     problem_statement = 'In base 13, what is 7a79 - -5?'
    #     f = extract_formal_elements(problem_statement)
    #     assert f == ['13', '7a79 - -5']
    #     assert eval_in_base(f[1], f[0]) == '7a81'
    #
    # def test_train_easy_arithmetic__nearest_integer_root_1(self):
    #     problem_statement = 'What is the square root of 664 to the nearest 1?'
    #     f = extract_formal_elements(problem_statement)
    #     root_f1 = root(f[1], f[0])
    #     rounded_root_f1 = round_to_int(root_f1, f[2])
    #     assert rounded_root_f1 == '26'
    #
    # def test_train_easy_arithmetic__nearest_integer_root_2(self):
    #     problem_statement = 'What is $f[1699] to the power of $f[1/6], to the nearest $f[1]?'
    #     f = extract_formal_elements(problem_statement)
    #     power_f0 = power(f[0], f[1])
    #     rounded_power_f0 = round_to_int(power_f0, f[2])
    #     assert rounded_power_f0 == '3'


