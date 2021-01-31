import re
import sympy as sym
from typing import List, Dict, Set
import multiprocess as mp
import time
# from math import log


# type definitions --------------------------------------


class Arb:
    """Arbitrary"""
    def __init__(self, arbitrary: str):
        self.arbitrary = str(arbitrary)

    def __str__(self):
        return self.arbitrary


class Eq(Arb):
    """Equation"""
    def __init__(self, equation: str):
        assert len(equation.split("=")) == 2
        self.equation = equation

    def __str__(self):
        return self.equation

    def __eq__(self, equation):
        return self.equation == str(equation)

    def split(self, split_on):
        return self.equation.split(split_on)


class Fn(Eq):
    """Function"""
    def __init__(self, function: str):
        assert len(function.split("=")) == 2
        function_arg_pattern = "([a-zA-Z0-9\s]+)\(([a-zA-Z0-9\s]+)\)"
        # extract parts of function definition
        lhs, rhs = function.split("=")
        match = re.match(function_arg_pattern, lhs)
        assert match is not None
        self.name, self.parameter = match.group(1), match.group(2)
        self.function = function

    def __str__(self):
        return str(self.function)

    def __eq__(self, function):
        return self.function == str(function)


class Ex(Arb):
    """Expression"""
    def __init__(self, expression: str):
        assert "=" not in expression
        self.expression = str(expression)

    def __str__(self):
        return self.expression

    def __eq__(self, other):
        return self.expression == other.expression

    def __hash__(self):
        return hash(self.expression)


class Var(Ex):
    """Variable"""
    def __init__(self, variable: str):
        self.variable = str(variable)
        assert variable.isalpha()

    def __str__(self):
        return self.variable

    def __eq__(self, variable):
        return self.variable == variable

    def __hash__(self):
        return hash(self.variable)


class Val(Ex):
    """Value"""
    def __init__(self, value: float):
        self.value = float(value)

    def __str__(self):
        if self.value % 1 == 0:
            return str(int(self.value))
        else:
            return str(self.value)

    def __eq__(self, value):
        return self.value == value.value

    def __hash__(self):
        return hash(str(self.value))

    def __lt__(self, other):
        return self.value < other.value

    def __gt__(self, other):
        return self.value > other.value

class Rat(Ex):
    def __init__(self, rational: str):
        self.rational = str(rational)
        self.numerator, self.denominator = [Val(x) for x in self.rational.split('/')]

    def __str__(self):
        return self.rational

    def __eq__(self, rational):
        return self.rational == str(rational)

# operator definitions --------------------------------------

# ss(system: List[Eq]) -> Dict[Var, Set[Val]]
def ss(system: list) -> dict:
    """
    solve_system
    solve a system of linear equations.

    :param system: List[
    :return: Dict[Var, Val]
    """
    def sympy_solve(system, return_dict):
        # run in try-except to suppress exception logging (must be done here due to use of multiprocess)
        try:
            solutions = sym.solve(system)
            return_dict["solutions"] = solutions
        except:
            pass

    sympy_equations = []
    for equation in system:
        lhs, rhs = str(equation).split("=")
        sympy_eq = sym.Eq(sym.sympify(lhs), sym.sympify(rhs))
        sympy_equations.append(sympy_eq)

    manager = mp.Manager()
    return_dict = manager.dict()
    p = mp.Process(target=sympy_solve, args=(sympy_equations, return_dict))
    p.start()
    p.join(1)

    if p.is_alive():
        p.terminate()
        p.join()
    solutions = return_dict.get("solutions", [])

    # Convert list to dictionary if no solution found.
    if len(solutions) == 0:
        raise Exception("no solution found")
    elif type(solutions) is dict:
        return {Var(str(k)): set([Val(float(v))]) for k, v in solutions.items()}
    elif type(solutions) is list:
        solutions_dict = {}
        for soln in solutions:
            for k, v in soln.items():
                if str(k) in solutions_dict.keys():
                    solutions_dict[Var(str(k))].add(Val(float(v)))
                else:
                    solutions_dict[Var(str(k))] = set([Val(float(v))])
        return solutions_dict


# ap(system: List[Eq], equation: Eq) -> List[Eq]
def ap(system: list, equation: Eq) -> list:
    """append"""
    if not system:
        return [equation]
    else:
        system.append(equation)
        return system


def ape(equation: Eq) -> list:
    """append_to_empty_list"""
    return [equation]


# lv(mapping: Dict[Var, Set[Val]], key: Var)
def lv(mapping: dict, key: Var) -> object:
    """lookup_value"""
    # TODO: figure out how to constrain output type in this case (multiple output types)
    assert key in mapping
    corresponding_set = mapping[key]
    if len(corresponding_set) == 1:
        return corresponding_set.pop()
    else:
        return corresponding_set


# lve(mapping: Dict[Var, Set[Val]], key: Var) -> Eq:
def lve(mapping: dict, key: Var) -> Eq:
    """lookup_value_equation"""
    assert key in mapping
    corresponding_set = mapping[key]
    value = corresponding_set.pop()
    return Eq(f"{key} = {value}")


def meq(expression1: Ex, expression2: Ex) -> Eq:
    """make equation"""
    return Eq(f"{expression1} = {expression2}")


def mfn(expression1: Ex, expression2: Ex) -> Fn:
    """make_function"""
    return Fn(f"{expression1} = {expression2}")


def eiv(equation: Eq) -> Var:
    """extract_isolated_variable"""
    lhs, rhs = str(equation).split("=")
    lhs, rhs = lhs.strip(), rhs.strip()
    if len(lhs) == 1 and lhs.isalpha():
        return lhs
    elif len(rhs) == 1 and rhs.isalpha():
        return rhs
    else:
        raise Exception("there is no isolated variable")


def pl(equation: Eq) -> Ex:
    """project_lhs"""
    return Ex(str(equation).split("=")[0].strip())


def pr(equation: Eq) -> Ex:
    """project_rhs"""
    return Ex(str(equation).split("=")[1].strip())


def slr(arb: Arb, eq: Eq) -> Arb:
    """substitution_left_to_right"""
    return Arb(str(arb).replace(str(pl(eq)), str(pr(eq))))


def srl(arb: Arb, eq: Eq) -> Arb:
    """substitution_right_to_left"""
    return Arb(str(arb).replace(str(pr(eq)), str(pl(eq))))


def fac(expression: Ex) -> Ex:
    """factor"""
    return Ex(str(sym.factor(expression)))


def sy(arb: Arb) -> Arb:
    """dimplify"""
    if "=" in str(arb):
        lhs, rhs = str(arb).split("=")
        lhs, rhs = lhs.strip(), rhs.strip()
        return Eq(f"{sym.simplify(lhs)} = {sym.simplify(rhs)}".strip())
    else:
        return Ex(str(sym.simplify(str(arb))).strip())


def df(expression: Ex) -> Ex:
    """diff"""
    return Ex(str(sym.diff(sym.sympify(str(expression)))))


def dfw(expression: Ex, variable: Var) -> Ex:
    """diff_wrt"""
    return Ex(str(sym.diff(sym.sympify(str(expression)), sym.sympify(str(variable)))))


def ra(function: Fn, var: Var) -> Fn:
    """replace_arg"""
    # TODO: make robust to longer names (e.g. max(x), y -> may(y) which is wrong)
    return Fn(str(function).replace(str(function.parameter), str(var)))


def mod(numerator: Val, discriminator: Val) -> Val:
    return Val(numerator.value % discriminator.value)


def md0(numerator: Val, discriminator: Val) -> bool:
    """mod_eq_0"""
    return numerator.value % discriminator.value == 0


def gcd(x: Val, y: Val) -> Val:
    from math import gcd

    return Val(gcd(int(x.value), int(y.value)))


def ip(x: Val) -> bool:
    """is_prime"""
    return sym.isprime(int(x.value))


def lcm(x: Val, y: Val) -> Val:
    import math
    assert int(x.value) == x.value and int(y.value) == y.value
    x, y = int(x.value), int(y.value)
    return Val(abs(x * y) // math.gcd(x, y))

def lcd(x: Rat, y: Rat) -> Val:
    "least common denominator"
    return lcm(x.denominator, y.denominator)


def pf(n: Val) -> set:
    """prime_factors"""
    # https://stackoverflow.com/questions/16996217/prime-factorization-list
    assert int(n.value) == n.value

    if ip(n):
        return n

    n = int(n.value)
    divisors = [d for d in range(2, n // 2 + 1) if n % d == 0]

    return set(
        [Val(d) for d in divisors if all(d % od != 0 for od in divisors if od != d)]
    )


def fa(
    function_definition: Fn, function_argument: Ex
) -> Val:
    """
    function_application
    :param function_definition: e.g. 'f(x) = x + x**3'
    :param function_argument: e.g. either '2' or 'f(2)'
    :return:
    """
    function_definition_pattern = "([a-zA-Z0-9\s]+)\(([a-zA-Z0-9\s]+)\)"
    function_arg_pattern = "([a-zA-Z0-9\s]+)\((-?[a-zA-Z0-9\s]+)\)"
    # extract parts of function definition
    lhs, rhs = str(function_definition).split("=")
    match = re.match(function_definition_pattern, lhs)
    function_name_from_definition, function_parameter = match.group(1), match.group(2)
    # extract parts of function argument
    function_argument_ = re.match(function_arg_pattern, str(function_argument))
    if function_argument_ is not None:
        function_name_from_argument, function_argument = (
            function_argument_.group(1),
            function_argument_.group(2),
        )
        assert function_name_from_definition == function_name_from_argument
    # evaluate function
    rhs_with_arg = rhs.replace(function_parameter, str(function_argument))
    return Val(eval(rhs_with_arg))


def nt(x: bool) -> bool:
    """not_op"""
    assert type(x) == bool
    return not x

