import unittest
import numpy as np
from environment.typed_operators import *


class Test(unittest.TestCase):
    def test_value(self):
        assert Val(1) == Val(1.0)
        assert {Val(1)} == {Val(1)}
        assert {Val(1)} == {Val(1.0)}

    def test_solve_system(self):
        system = [Eq("x = 1")]
        assert ss(system) == {Var("x"): {Val(1)}}

        system = [Eq("x = 1"), Eq("y = 1")]
        assert ss(system) == {
            Var("x"): {Val(1)},
            Var("y"): {Val(1)},
        }

        system = [Eq("x + y = 1"), Eq("x - y = 1")]
        assert ss(system) == {
            Var("x"): {Val(1)},
            Var("y"): {Val(0)},
        }

        system = [Eq("3*x + y = 9"), Eq("x + 2*y = 8")]
        assert ss(system) == {
            Var("x"): {Val(2)},
            Var("y"): {Val(3)},
        }

        # # fails on singular matrix
        system = [
            Eq("x + 2*y - 3*z = 1"),
            Eq("3*x - 2*y + z = 2"),
            Eq("-x + 2*y - 2*z = 3"),
        ]
        self.assertRaises(Exception, ss, system)

        # system with floating point coefficients
        system = [Eq("-15 = 3*c + 2.0*c")]
        assert ss(system) == {Var("c"): {Val(-3)}}

        # quadratic equation
        system = [Eq("-3*h**2/2 - 24*h - 45/2 = 0")]
        assert ss(system) == {Var("h"): {Val(-15.0), Val(-1.0)}}

        # unsolvable equation / infinite loop without timeout
        system = [Eq('-4*i**3*j**3 - 2272*i**3 - 769*i**2*j - j**3 = 1')]
        self.assertRaises(Exception, ss, system)

        system = [Eq('-g**3 - 9*g**2 - g + l(g) - 10 = 0')]
        self.assertRaises(Exception, ss, system)

    def test_is_prime(self):
        assert ip(Val('19373'))
        assert nt(ip(Val('19374')))

    def test_prime_factors(self):
        result = pf(Val('7380'))
        assert ", ".join([str(x) for x in sorted(list(result))]) == '2, 3, 5, 41'

    def test_lcd(self):
        assert lcd(Rat('2/3'), Rat('3/5')) == Val('15')
