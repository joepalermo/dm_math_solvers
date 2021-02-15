import unittest
import numpy as np
from environment.typed_operators import *
from sympy import sympify


class Test(unittest.TestCase):
    def test_value(self):
        assert Value(1) == Value(1.0)
        assert {Value(1)} == {Value(1)}
        assert {Value(1)} == {Value(1.0)}

    def test_solve_system(self):
        system = [Equation("x = 1")]
        assert solve_system(system) == {Variable("x"): {Value(1)}}

        system = [Equation("x = 1"), Equation("y = 1")]
        assert solve_system(system) == {
            Variable("x"): {Value(1)},
            Variable("y"): {Value(1)},
        }

        system = [Equation("x + y = 1"), Equation("x - y = 1")]
        assert solve_system(system) == {
            Variable("x"): {Value(1)},
            Variable("y"): {Value(0)},
        }

        system = [Equation("3*x + y = 9"), Equation("x + 2*y = 8")]
        assert solve_system(system) == {
            Variable("x"): {Value(2)},
            Variable("y"): {Value(3)},
        }

        # # fails on singular matrix
        system = [
            Equation("x + 2*y - 3*z = 1"),
            Equation("3*x - 2*y + z = 2"),
            Equation("-x + 2*y - 2*z = 3"),
        ]
        self.assertRaises(Exception, solve_system, system)

        # system with floating point coefficients
        system = [Equation("-15 = 3*c + 2.0*c")]
        assert solve_system(system) == {Variable("c"): {Value(-3)}}

        # quadratic equation
        system = [Equation("-3*h**2/2 - 24*h - 45/2 = 0")]
        assert solve_system(system) == {Variable("h"): {Value(-15.0), Value(-1.0)}}

        # unsolvable equation / infinite loop without timeout
        system = [Equation('-4*i**3*j**3 - 2272*i**3 - 769*i**2*j - j**3 = 1')]
        self.assertRaises(Exception, solve_system, system)

        system = [Equation('-g**3 - 9*g**2 - g + l(g) - 10 = 0')]
        self.assertRaises(Exception, solve_system, system)

        # unsolvable equation / infinite loop without timeout
        system = [Equation('-4*i**3*j**3 - 2272*i**3 - 769*i**2*j - j**3 = 1')]
        self.assertRaises(Exception, solve_system, system)

    def test_is_prime(self):
        assert is_prime(Value('19373'))
        assert not_op(is_prime(Value('19374')))

    def test_prime_factors(self):
        result = prime_factors(Value('7380'))
        assert ", ".join([str(x) for x in sorted(list(result))]) == '2, 3, 5, 41'

    def test_lcd(self):
        assert lcd(Rational('2/3'), Rational('3/5')) == Value('15')
        assert lcd(Rational('2/3'), Rational('3/5')) == Value('15')

    def test_third_derivative(self):
        inpt = Expression('-272*j**5 + j**3 - 8234*j**2')
        third_derivative = differentiate(differentiate(differentiate(inpt)))
        assert sympify(third_derivative) == sympify(Expression('-16320*j**2 + 6'))

    def test_simplify_not_identity(self):
        inpt = Expression('2*x')
        output = simplify(inpt)
        assert output is None

    def test_factor_not_identity(self):
        inpt = Expression('2*x')
        output = factor(inpt)
        assert output is None
