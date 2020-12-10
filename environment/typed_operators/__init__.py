import re
import sympy as sym
from typing import List, Dict, Set
# from math import log


# type definitions --------------------------------------


class Arbitrary:
    def __init__(self, arbitrary: str):
        self.arbitrary = str(arbitrary)

    def __str__(self):
        return self.arbitrary


class Equation(Arbitrary):
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
    def __init__(self, function: str):
        assert len(function.split('=')) == 2
        function_arg_pattern = '([a-zA-Z0-9\s]+)\(([a-zA-Z0-9\s]+)\)'
        # extract parts of function definition
        lhs, rhs = function.split('=')
        match = re.match(function_arg_pattern, lhs)
        assert match is not None
        self.name, self.parameter = match.group(1), match.group(2)
        self.function = function

    def __str__(self):
        return str(self.function)

    def __eq__(self, function):
        return self.function == str(function)

class Expression(Arbitrary):
    def __init__(self, expression: str):
        assert '=' not in expression
        self.expression = str(expression)

    def __str__(self):
        return self.expression

    def __eq__(self, other):
        return self.expression == other.expression

    def __hash__(self):
        return hash(self.expression)


class Variable(Expression):
    def __init__(self, variable: str):
        self.variable = str(variable)
        assert variable.isalpha()

    def __str__(self):
        return self.variable

    def __eq__(self, variable):
        return self.variable == variable

    def __hash__(self):
        return hash(self.variable)


class Value(Expression):
    def __init__(self, value: float):
        self.value = float(value)

    def __str__(self):
        return str(self.value)

    def __eq__(self, value):
        return self.value == value.value

    def __hash__(self):
        return hash(str(self.value))
  
    
# operator definitions --------------------------------------

# solve_system(system: List[Equation]) -> Dict[Variable, Set[Value]]
def solve_system(system: list) -> dict:
    '''
    solve a system of linear equations.

    :param system: List[
    :return: Dict[Variable, Value]
    '''
    sympy_equations = []
    for equation in system:
        lhs, rhs = str(equation).split('=')
        sympy_eq = sym.Eq(sym.sympify(lhs), sym.sympify(rhs))
        sympy_equations.append(sympy_eq)
    solutions = sym.solve(sympy_equations)
    # Convert list to dictionary if no solution found.
    if len(solutions) == 0:
        raise Exception("no solution found")
    elif type(solutions) is dict:
        return {Variable(str(k)): set([Value(float(v))]) for k,v in solutions.items()}
    elif type(solutions) is list:
        solutions_dict = {}
        for soln in solutions:
            for k, v in soln.items():
                if str(k) in solutions_dict.keys():
                    solutions_dict[Variable(str(k))].add(Value(float(v)))
                else:
                    solutions_dict[Variable(str(k))] = set([Value(float(v))])
        return solutions_dict

# append(system: List[Equation], equation: Equation) -> List[Equation]
def append(system: list, equation: Equation) -> list:
    if not system:
        return [equation]
    else:
        system.append(equation)
        return system

# lookup_value(mapping: Dict[Variable, Set[Value]], key: Variable)
def lookup_value(mapping: dict, key: Variable) -> object:
    # TODO: figure out how to constrain output type in this case (multiple output types)
    assert key in mapping
    corresponding_set = mapping[key]
    if len(corresponding_set) == 1:
        return corresponding_set.pop()
    else:
        return corresponding_set

# lookup_value_eq(mapping: Dict[Variable, Set[Value]], key: Variable) -> Equation:
def lookup_value_eq(mapping: dict, key: Variable) -> Equation:
    assert key in mapping
    corresponding_set = mapping[key]
    value = corresponding_set.pop()
    return Equation(f"{key} = {value}")


def make_equality(expression1: Expression, expression2: Expression) -> Equation:
    return Equation(f"{expression1} = {expression2}")

def make_function(expression1: Expression, expression2: Expression) -> Function:
    return Function(f"{expression1} = {expression2}")


def extract_isolated_variable(equation: Equation) -> Variable:
    lhs, rhs = str(equation).split('=')
    lhs, rhs = lhs.strip(), rhs.strip()
    if len(lhs) == 1 and lhs.isalpha():
        return lhs
    elif len(rhs) == 1 and rhs.isalpha():
        return rhs
    else:
        raise Exception("there is no isolated variable")


def project_lhs(equation: Equation) -> Expression:
    return Expression(str(equation).split('=')[0].strip())


def project_rhs(equation: Equation) -> Expression:
    return Expression(str(equation).split('=')[1].strip())


def substitution_left_to_right(arb: Arbitrary, eq: Equation) -> Arbitrary:
    return Arbitrary(str(arb).replace(str(project_lhs(eq)), str(project_rhs(eq))))


def substitution_right_to_left(arb: Arbitrary, eq: Equation) -> Arbitrary:
    return Arbitrary(str(arb).replace(str(project_rhs(eq)), str(project_lhs(eq))))


def factor(expression: Expression) -> Expression:
    return Expression(str(sym.factor(expression)))

def simplify(arb: Arbitrary) -> Arbitrary:
    if '=' in str(arb):
        lhs, rhs = str(arb).split('=')
        lhs, rhs = lhs.strip(), rhs.strip()
        return Equation(f'{sym.simplify(lhs)} = {sym.simplify(rhs)}'.strip())
    else:
        return Expression(str(sym.simplify(str(arb))).strip())

def diff(expression: Expression) -> Expression:
    return Expression(str(sym.diff(sym.sympify(str(expression)))))

def replace_arg(function: Function, var: Variable) -> Function:
    # TODO: make robust to longer names (e.g. max(x), y -> may(y) which is wrong)
    return Function(str(function).replace(str(function.parameter), str(var)))

# def max_arg(x: Expression, y: Expression) -> Expression:
#     return str(max(eval(str(x)),eval(str(y))))
