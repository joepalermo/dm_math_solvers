import unittest
import numpy as np
from environment.operators.solve_linsys import extract_coefficients, solve_linsys
from environment.operators import add_key_pair


class Test(unittest.TestCase):

    def test_extract_coefficients(self):
        # signs
        assert extract_coefficients('x=y') == {'x': 1, 'y': -1}
        assert extract_coefficients('-x=y') == {'x': -1, 'y': -1}
        assert extract_coefficients('x=-y') == {'x': 1, 'y': 1}
        assert extract_coefficients('-x=-y') == {'x': -1, 'y': 1}

        # spaces
        assert extract_coefficients(' x = y ') == {'x': 1, 'y': -1}
        assert extract_coefficients(' - x = y ') == {'x': -1, 'y': -1}
        assert extract_coefficients(' x = - y ') == {'x': 1, 'y': 1}
        assert extract_coefficients(' - x = - y ') == {'x': -1, 'y': 1}

        # coefficients
        assert extract_coefficients('2*x=y') == {'x': 2, 'y': -1}
        assert extract_coefficients('-2*x=3*y') == {'x': -2, 'y': -3}
        assert extract_coefficients('7*x=-3*y') == {'x': 7, 'y': 3}
        assert extract_coefficients('-10*x=-11*y') == {'x': -10, 'y': 11}

        # coefficients with null
        assert extract_coefficients('2*x=y+5') == {'x': 2, 'y': -1, None: 5}
        assert extract_coefficients('-2*x=3*y-1') == {'x': -2, 'y': -3, None: -1}
        assert extract_coefficients('7*x=-3*y+11') == {'x': 7, 'y': 3, None: 11}
        assert extract_coefficients('-10*x=-11*y-5') == {'x': -10, 'y': 11, None: -5}

        # multiple of same coefficients
        assert extract_coefficients('2*x + 1 = - x + y - 4') == {'x': 3, 'y': -1, None: -5}

    def test_solve_linsys(self):
        system = 'x = 1'
        assert solve_linsys(system) == {'x': 1}

        system = 'x + 1 = 0'
        assert solve_linsys(system) == {'x': -1}

        system = 'x = -x'
        assert solve_linsys(system) == {'x': 0}

        system = 'x = 2*x + 1'
        assert solve_linsys(system) == {'x': -1}

        system = ['x = 1', 'y = 1']
        assert solve_linsys(system) == {'x': 1, 'y': 1}

        system = ['x + y = 1', 'x - y = 1']
        assert solve_linsys(system) == {'x': 1, 'y': 0}

        system = ['3*x + y = 9', 'x + 2*y = 8']
        assert solve_linsys(system) == {'x': 2, 'y': 3}

        # fails on singular matrix
        system = ['x + 2*y - 3*z = 1', '3*x - 2*y + z = 2', '-x + 2*y - 2*z = 3']
        self.assertRaises(np.linalg.LinAlgError, solve_linsys, system)

        system = ['0 = 4*b + b + 15']
        assert solve_linsys(system) == {'b': -3}

        system = ['0 = 4*f - 0*t - 4*t - 4', '-4*f + t = -13']
        assert solve_linsys(system) == {'t': 3.0, 'f': 4.0}

    def test_key_pair(self):
        assert add_key_pair(None, 'x', 2) == {'x': 2}
        assert add_key_pair(add_key_pair(None, 'x', 2), 'y', 3) == {'x': 2, 'y': 3}
