import unittest
import numpy as np
from environment.operators import add_keypair, simplify, solve_system, eval_in_base, round_to_int


class Test(unittest.TestCase):

    def test_solve_system(self):
        system = 'x = 1'
        assert solve_system(system) == {'x': 1}

        system = 'x + 1 = 0'
        assert solve_system(system) == {'x': -1}

        system = 'x = -x'
        assert solve_system(system) == {'x': 0}

        system = 'x = 2*x + 1'
        assert solve_system(system) == {'x': -1}

        system = ['x = 1', 'y = 1']
        assert solve_system(system) == {'x': 1, 'y': 1}

        system = ['x + y = 1', 'x - y = 1']
        assert solve_system(system) == {'x': 1, 'y': 0}

        system = ['3*x + y = 9', 'x + 2*y = 8']
        assert solve_system(system) == {'x': 2, 'y': 3}

        # # fails on singular matrix
        system = ['x + 2*y - 3*z = 1', '3*x - 2*y + z = 2', '-x + 2*y - 2*z = 3']
        self.assertRaises(Exception, solve_system, system)

        system = '0 = 4*b + b + 15'
        assert solve_system(system) == {'b': -3}

        system = ['0 = 4*f - 0*t - 4*t - 4', '-4*f + t = -13']
        assert solve_system(system) == {'t': 3.0, 'f': 4.0}

        # system with floating point coefficients
        system = '-15 = 3*c + 2.0*c'
        assert solve_system(system) == {'c': -3}

        # quadratic equation
        system = '-3*h**2/2 - 24*h - 45/2 = 0'
        assert solve_system(system) == {'h': {-15.0, -1.0}}

    def test_key_pair(self):
        assert add_keypair(None, 'x', 2) == {'x': 2}
        assert add_keypair(add_keypair(None, 'x', 2), 'y', 3) == {'x': 2, 'y': 3}

    def test_simplify(self):
        assert simplify('x + 1 + 1') == 'x + 2'
        assert simplify('b = w - -6') == 'b = w + 6'

    def test_eval_in_base(self):
        assert eval_in_base('7a79 - -5', '13') == '7a81'
        assert eval_in_base('-100 + -1001100', '2') == '-1010000'
        assert eval_in_base('-5aa - 8', '12') == '-5b6'

    def test_round_to_int(self):
        assert round_to_int('123456', '1000000') == '0'
        assert round_to_int('123456', '100000') == '100000'
        assert round_to_int('123456', '10000') == '120000'
        assert round_to_int('123456', '1000') == '123000'
        assert round_to_int('123456', '100') == '123500'
        assert round_to_int('123456', '10') == '123460'
        assert round_to_int('123456', '1') == '123456'
        assert round_to_int('123456', '5') == '123455'
        assert round_to_int('123453', '5') == '123455'