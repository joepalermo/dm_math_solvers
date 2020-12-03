import unittest
import numpy as np
from environment.operators import add_keypair, simplify, solve_system


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
        assert solve_system(system) == [{'h': -15}, {'h': -1}]

    def test_key_pair(self):
        assert add_keypair(None, 'x', 2) == {'x': 2}
        assert add_keypair(add_keypair(None, 'x', 2), 'y', 3) == {'x': 2, 'y': 3}

    def test_simplify(self):
        assert simplify('x + 1 + 1') == 'x + 2'
        assert simplify('b = w - -6') == 'b = w + 6'