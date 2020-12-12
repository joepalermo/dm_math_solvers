import unittest
import numpy as np
from environment.typed_operators import *

class Test(unittest.TestCase):

    def test_value(self):
        assert Value(1) == Value(1.0)
        assert {Value(1)} == {Value(1)}
        assert {Value(1)} == {Value(1.0)}

    def test_solve_system(self):
        system = [Equation('x = 1')]
        assert solve_system(system) == {Variable('x'): {Value(1)}}

        system = [Equation('x = 1'), Equation('y = 1')]
        assert solve_system(system) == {Variable('x'): {Value(1)}, Variable('y'): {Value(1)}}

        system = [Equation('x + y = 1'), Equation('x - y = 1')]
        assert solve_system(system) == {Variable('x'): {Value(1)}, Variable('y'): {Value(0)}}

        system = [Equation('3*x + y = 9'), Equation('x + 2*y = 8')]
        assert solve_system(system) == {Variable('x'): {Value(2)}, Variable('y'): {Value(3)}}

        # # fails on singular matrix
        system = [Equation('x + 2*y - 3*z = 1'), Equation('3*x - 2*y + z = 2'), Equation('-x + 2*y - 2*z = 3')]
        self.assertRaises(Exception, solve_system, system)

        # system with floating point coefficients
        system = [Equation('-15 = 3*c + 2.0*c')]
        assert solve_system(system) == {Variable('c'): {Value(-3)}}

        # quadratic equation
        system = [Equation('-3*h**2/2 - 24*h - 45/2 = 0')]
        assert solve_system(system) == {Variable('h'): {Value(-15.0), Value(-1.0)}}

