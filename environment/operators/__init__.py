import re

def append(ls, value):
    if not ls:
        return [value]
    else:
        ls.append(value)
        return ls


def add_keypair(mapping, key, value):
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


print(function_application('f(x) = x + x**3', 'f(2)'))

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