import re
import sympy as sym


def append(ls, value):
    if not ls:
        return [value]
    else:
        ls.append(value)
        return ls


def add_keypair(mapping, key, value=None):
    if value is None:
        key, value = key.split('=')
        key, value = key.strip(), value.strip()
    if not mapping:
        return {key: value}
    else:
        mapping[key] = value
        return mapping


def lookup_value(mapping, key):
    return mapping.get(key, None)


def function_application(function_definition, function_argument):
    '''

    :param function_definition: e.g. 'f(x) = x + x**3'
    :param function_argument: e.g. either '2' or 'f(2)'
    :return:
    '''
    function_arg_pattern = '([a-zA-Z0-9\s]+)\(([a-zA-Z0-9\s]+)\)'
    # extract parts of function definition
    lhs, rhs = function_definition.split('=')
    match = re.match(function_arg_pattern, lhs)
    function_name_from_definition, function_parameter = match.group(1), match.group(2)
    # extract parts of function argument
    function_argument_ = re.match(function_arg_pattern, function_argument)
    if function_argument_ is not None:
        function_name_from_argument, function_argument = function_argument_.group(1), function_argument_.group(2)
        assert function_name_from_definition == function_name_from_argument
    # evaluate function
    rhs_with_arg = rhs.replace(function_parameter, str(function_argument))
    return eval(rhs_with_arg)


def apply_mapping(expression, mapping):
    for key, value in mapping.items():
        expression = expression.replace(key, str(value))
    return expression


def make_equality(expression1, expression2):
    return f"{expression1} = {expression2}"


def project_lhs(eq):
    return eq.split('=')[0].strip()


def project_rhs(eq):
    return eq.split('=')[1].strip()


def simplify(expression):
    if '=' in expression:
        lhs, rhs = expression.split('=')
        lhs, rhs = lhs.strip(), rhs.strip()
        return f'{sym.simplify(lhs)} = {sym.simplify(rhs)}'.strip()
    else:
        return str(sym.simplify(expression)).strip()


def solve_system(system):
    '''
    solve a system of linear equations.

    :param system: either a string or a list of strings
    :return: dict mapping variable names to values
    '''
    if type(system) == str:
        system = [system]
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
        return  {str(k): v for k,v in solutions.items()}
    elif type(solutions) is list:
        solutions_dict = {}
        for soln in solutions:
            for k, v in soln.items():
                if str(k) in solutions_dict.keys():
                    solutions_dict[str(k)].add(float(v))
                else:
                    solutions_dict[str(k)] = set([float(v)])
        return solutions_dict


def replace_arg(function, var):
    lhs, rhs = project_lhs(function), project_rhs(function)
    #Find argument to replace:
    arg_pattern = '\(([a-zA-Z0-9]+)\)'
    arg = re.findall(arg_pattern, lhs)[0]
    return make_equality(lhs.replace(arg,var), rhs.replace(arg,var))


def substitution(exp_or_eq, eq2): #Left to right substitution
    return exp_or_eq.replace(project_lhs(eq2), project_rhs(eq2))

# arithmetic -------------------------------------

def add(x, y):
    return x + y


def sub(x, y):
    return x - y


def mult(x, y):
    return x * y


def rational_div(x, y):
    '''rational number division'''
    from fractions import Fraction
    return Fraction(x) / Fraction(y)


def calc(expression):
    return eval(expression)


def gcd(x, y):
    from math import gcd
    return gcd(x, y)


def factor(expression):
    return str(sym.factor(expression))


def diff(expression):
    return str(sym.diff(sym.sympify(expression)))


class NumberInBase(object):
  """Contains value, represented in a given base."""

  def __init__(self, value, base):
    """Initializes a `NumberInBase`.
    Args:
      value: Positive or negative integer.
      base: Integer in the range [2, 36].
    Raises:
      ValueError: If base is not in the range [2, 36] (since this is the limit
          that can be represented by 10 numbers plus 26 letters).
    """
    if not 2 <= base <= 36:
      raise ValueError('base={} must be in the range [2, 36]'.format(base))
    self._value = value
    self._base = base

    chars = []
    remainder = abs(value)
    while True:
      digit = remainder % base
      char = str(digit) if digit <= 9 else chr(ord('a') + digit - 10)
      chars.append(char)
      remainder = int(remainder / base)
      if remainder == 0:
        break
    if value < 0:
      chars.append('-')

    self._str = ''.join(reversed(chars))

  def __str__(self):
    return self._str

  def _sympy_(self):
    return self._value


def eval_in_base(exp, base):
    base = int(base)
    matching_regions = list()
    for match in re.finditer('[a-zA-Z0-9]+', exp):
        start, end = match.start(0), match.end(0)
        matching_regions.append((start, end))
    numbers = [exp[start:end] for start, end in matching_regions]
    numbers_in_decimal = [str(int(num, base)) for num in numbers]
    prev_end = 0
    new_exp = ''
    for (start, end), dec_num in zip(matching_regions, numbers_in_decimal):
        new_exp += exp[prev_end:start] + dec_num
        prev_end = end
    new_exp += exp[prev_end:]
    return str(NumberInBase(eval(new_exp), base))


def round(num, round_to):
    pass