# import re
import sympy as sym
from typing import List, Dict, Set
# from math import log


class Equation:
    def __init__(self, equation: str):
        assert len(equation.split('=')) == 2
        self.equation = equation

    def __str__(self):
        return self.equation

    def __eq__(self, equation):
        return self.equation == str(equation)

    def split(self, split_on):
        return self.equation.split(split_on)


class Function(Equation):
    pass


class Expression:
    def __init__(self, expression: str):
        self.expression = expression


class Variable(Expression):
    def __init__(self, variable: str):
        assert variable.isalpha()
        self.variable = variable

    def __eq__(self, variable):
        return self.variable == variable

    def __hash__(self):
        return hash(self.variable)


class Value(Expression):
    def __init__(self, value: float):
        self.value = value

    def __str__(self):
        return self.value

    def __eq__(self, value):
        return self.value == value

    def __hash__(self):
        return hash(str(self.value))


def append(system: List[Equation], equation: Equation) -> List[Equation]:
    if not system:
        return [equation]
    else:
        system.append(equation)
        return system


def solve_system(system: List[Equation]) -> Dict[Variable, Set[Value]]:
    '''
    solve a system of linear equations.

    :param system: List[
    :return: Dict[Variable, Value]
    '''
    sympy_equations = []
    for equation in system:
        lhs, rhs = equation.split('=')
        sympy_eq = sym.Eq(sym.sympify(lhs), sym.sympify(rhs))
        sympy_equations.append(sympy_eq)
    solutions = sym.solve(sympy_equations)
    # Convert list to dictionary if no solution found.
    if len(solutions) == 0:
        raise Exception("no solution found")
    elif type(solutions) is dict:
        return {str(k): set([Value(float(v))]) for k,v in solutions.items()}
    elif (type(solutions) is list) or (type(solutions) is dict):
        solutions_dict = {}
        for soln in solutions:
            for k, v in soln.items():
                if str(k) in solutions_dict.keys():
                    solutions_dict[str(k)].add(Value(float(v)))
                else:
                    solutions_dict[str(k)] = set([Value(float(v))])
        return solutions_dict


def lookup_value(mapping: Dict[Variable, Set[Value]], key: Variable) -> Value:
    assert key in mapping
    corresponding_set = mapping[key]
    value = corresponding_set.pop()
    return value


def lookup_value_eq(mapping: Dict[Variable, Set[Value]], key: Variable) -> Equation:
    assert key in mapping
    corresponding_set = mapping[key]
    value = corresponding_set.pop()
    return Equation(f"{key} = {value}")
