import unittest
from environment.utils import extract_formal_elements, cast_formal_element
from environment.typed_operators import *


class Test(unittest.TestCase):
    """all test cases are taken from train-easy"""

    def test_easy_algebra__linear_1d(self):
        question = "Solve 0 = 4*b + b + 15 for b."
        fs = extract_formal_elements(question)
        assert fs == [Eq("0 = 4*b + b + 15"), Var("b")]
        system = ap([], fs[0])
        solution = ss(system)
        value = lv(solution, fs[1])
        assert value == Val(-3)

    def test_easy_algebra__linear_1d_composed(self):
        question = "Let w be (-1 + 13)*3/(-6). Let b = w - -6. Let i = 2 - b. Solve -15 = 3*c + i*c for c."
        f = extract_formal_elements(question)
        assert f == [
            Var("w"),
            Ex("(-1 + 13)*3/(-6)"),
            Eq("b = w - -6"),
            Eq("i = 2 - b"),
            Eq("-15 = 3*c + i*c"),
            Var("c"),
        ]
        eq1 = meq(f[0], f[1])
        system = ap(ap(ap([], eq1), f[2]), f[3])
        soln = ss(system)
        i_eq = lve(soln, eiv(f[3]))
        lin_eq = slr(f[4], i_eq)
        assert lv(ss(ap([], lin_eq)), f[5]) == Val(-3)

    def test_easy_algebra__linear_2d(self):
        question = "Solve 0 = 4*f - 0*t - 4*t - 4, -4*f + t = -13 for f."
        f = extract_formal_elements(question)
        assert f == [
            Eq("0 = 4*f - 0*t - 4*t - 4"),
            Eq("-4*f + t = -13"),
            Var("f"),
        ]

        assert lv(
            ss(ap(ap([], f[0]), f[1])), f[2]
        ) == Val(4)

    def test_algebra__linear_2d_composed(self):
        question = "Suppose 2*y + 12 = 6*y. Suppose y = f - 15. Solve -8 = -4*w, -3*d - 4*w + f = -8*d for d."
        f = extract_formal_elements(question)
        assert f == [
            Eq("2*y + 12 = 6*y"),
            Eq("y = f - 15"),
            Eq("-8 = -4*w"),
            Eq("-3*d - 4*w + f = -8*d"),
            Var("d"),
        ]
        system = ap(ap(ap(ap(None, f[0]), f[1]), f[2]), f[3])
        assert lv(ss(system), f[4]) == Val(-2)

    def test_algebra__polynomial_roots_1(self):
        question = "Solve -3*h**2/2 - 24*h - 45/2 = 0 for h."
        f = extract_formal_elements(question)
        assert f == [Eq("-3*h**2/2 - 24*h - 45/2 = 0"), Var("h")]
        soln = lv(ss(ap([], f[0])), f[1])
        assert soln == {Val(-1), Val(-15)}

    def test_algebra__polynomial_roots_2(self):
        question = "Factor -n**2/3 - 25*n - 536/3."
        f = extract_formal_elements(question)
        assert f == [Ex("-n**2/3 - 25*n - 536/3")]
        assert fac(f[0]) == Ex("-(n + 8)*(n + 67)/3")

    def test_algebra__polynomial_roots_3(self):
        question = (
            "Find s such that 9*s**4 - 8958*s**3 - 14952*s**2 - 2994*s + 2991 = 0."
        )
        f = extract_formal_elements(question)
        assert f == [
            Var("s"),
            Eq("9*s**4 - 8958*s**3 - 14952*s**2 - 2994*s + 2991 = 0"),
        ]
        assert lv(ss(ap([], f[1])), f[0]) == {
            Val(-1),
            Val(1 / 3),
            Val(997),
        }

    def test_algebra__polynomial_roots_composed_1(self):
        question = "Let d = -25019/90 - -278. Let v(j) be the third derivative of 0 + 1/27*j**3 - d*j**5 + 1/54*j**4 + 3*j**2 + 0*j. Suppose v(o) = 0. What is o?"
        f = extract_formal_elements(question)
        assert f == [
            Eq("d = -25019/90 - -278"),
            Ex("v(j)"),
            Ex("0 + 1/27*j**3 - d*j**5 + 1/54*j**4 + 3*j**2 + 0*j"),
            Fn("v(o) = 0"),
            Var("o"),
        ]
        d = sy(f[0])
        function = slr(f[2], d)
        v = df(df(df(function)))
        v_eq = mfn(f[1], v)
        v_eq_o = ra(v_eq, f[4])
        equation = slr(
            f[3], v_eq_o
        )  # e.g. x.subs(sym.sympify('f(x)'), sym.sympify('v'))
        assert lv(ss(ap([], equation)), f[4]) == {
            Val(-1 / 3),
            Val(1),
        }

    def test_calculus__differentiate(self):
        question = "What is the second derivative of 2*c*n**2*z**3 + 30*c*n**2 + 2*c*n*z**2 - 2*c + n**2*z**2 - 3*n*z**3 - 2*n*z wrt n?"
        f = extract_formal_elements(question)
        assert f == [Ex('2*c*n**2*z**3 + 30*c*n**2 + 2*c*n*z**2 - 2*c + n**2*z**2 - 3*n*z**3 - 2*n*z'), Var('n')]
        assert dfw(dfw(f[0], f[1]), f[1]) == Ex('4*c*z**3 + 60*c + 2*z**2')

    def test_numbers__div_remainder(self):
        question = "Calculate the remainder when 93 is divided by 59."
        f = extract_formal_elements(question)
        assert f == [Val("93"), Val("59")]
        assert mod(f[0], f[1]) == Val("34")

    def test_numbers__gcd(self):
        question = "Calculate the greatest common fac of 11130 and 6."
        f = extract_formal_elements(question)
        assert f == [Val("11130"), Val("6")]
        assert gcd(f[0], f[1]) == Val("6")

    def test_numbers__is_factor(self):
        question = "Is 15 a fac of 720?"
        f = extract_formal_elements(question)
        assert f == [Val("15"), Val("720")]
        assert md0(f[1], f[0]) == True

    def test_numbers__is_prime(self):
        question = "Is 93163 a prime number?"
        f = extract_formal_elements(question)
        assert f == [Val("93163")]
        assert ip(f[0]) == False

    def test_numbers__lcm(self):
        question = "Calculate the smallest common multiple of 351 and 141."
        f = extract_formal_elements(question)
        assert f == [Val("351"), Val("141")]
        assert lcm(f[0], f[1]) == Val("16497")

    def test_numbers__list_prime_factors(self):
        question = "What are the prime factors of 329?"
        f = extract_formal_elements(question)
        assert f == [Val("329")]
        assert pf(f[0]) == {Val(7), Val(47)}

    def test_polynomials_evaluate(self):
        question = "Let i(h) = -7*h - 15. Determine i(-2)."
        f = extract_formal_elements(question)
        assert f == [Fn("i(h) = -7*h - 15"), Ex("i(-2)")]
        assert fa(f[0], f[1]) == Val(-1)

    #  requiring new operators --------------------------------------------

    # def test_comparison__closest(self):
    #     question = 'Which is the closest to -1/3?  (a) -8/7  (b) 5  (c) -1.3'
    #     f = extract_formal_elements(question)
    #     power_f0 = power(f[0], f[1])
    #     rounded_power_f0 = round_to_int(power_f0, f[2])
    #     assert rounded_power_f0 == '3'

    # def test_comparison__pair_composed(self):
    #     question = 'Let o = -788/3 - -260. Which is bigger: -0.1 or o?'
    #     f = extract_formal_elements(question)
    #     assert f == [Eq('o = -788/3 - -260'), Val('-0.1'), Var('o')]
    #     o = sy(f[0])
    #     m = max_arg(f[1], pr(o))
    #     assert srl(m, o) == Val('-0.1')

    # def test_comparison__sort_composed(self):
    #     question = 'Suppose $f[0 = -4*x + 8*x - 40]. Let $f[h(i) = i**2 - 9*i - 14]. Let $f[n] be $f[h(x)]. Sort $f[-1], $f[4], $f[n].'
    #     f = extract_formal_elements(question)
    #     x = lv(ss(f[0]), get

    # def test_arithmetic__add_or_sub_in_base(self):
    #     question = 'In base 13, what is 7a79 - -5?'
    #     f = extract_formal_elements(question)
    #     assert f == ['13', '7a79 - -5']
    #     assert eval_in_base(f[1], f[0]) == '7a81'
    #
    # def test_arithmetic__nearest_integer_root_1(self):
    #     question = 'What is the square root of 664 to the nearest 1?'
    #     f = extract_formal_elements(question)
    #     root_f1 = root(f[1], f[0])
    #     rounded_root_f1 = round_to_int(root_f1, f[2])
    #     assert rounded_root_f1 == '26'
    #
    # def test_arithmetic__nearest_integer_root_2(self):
    #     question = 'What is $f[1699] to the power of $f[1/6], to the nearest $f[1]?'
    #     f = extract_formal_elements(question)
    #     power_f0 = power(f[0], f[1])
    #     rounded_power_f0 = round_to_int(power_f0, f[2])
    #     assert rounded_power_f0 == '3'
