import re
import numpy as np

# in order from operators_with_graphs

def extract_coefficients(linear_equation):
    '''
    :param linear_equation:
    :return: dict mapping symbol to coefficient
    '''
    def extract_signs(expression):
        signs = list()
        for token in expression:
            if token == '-':
                signs.append('-')
            elif token == '+':
                signs.append('+')
        return signs
    linear_equation = linear_equation.strip()
    # extract signs of each component
    lhs, rhs = linear_equation.split('=')
    lhs = lhs.strip()
    rhs = rhs.strip()
    lhs_first_sign = '-' if lhs[0] == '-' else '+'
    rhs_first_sign = '-' if rhs[0] == '-' else '+'
    rest_of_lhs_signs = extract_signs(lhs[1:])
    rest_of_rhs_signs = extract_signs(rhs[1:])
    lhs_signs = [lhs_first_sign] + rest_of_lhs_signs
    rhs_signs = [rhs_first_sign] + rest_of_rhs_signs
    signs = lhs_signs + rhs_signs
    # split out sign information and extract cofficients for variables
    components = [c for c in re.split('\+|-|=', linear_equation) if not c.isspace() and len(c) > 0]
    assert len(signs) == len(components)
    coefficients = dict()
    for i, component in enumerate(components):
        split_component = component.split('*')
        split_component = [c.strip() for c in split_component]
        if len(split_component) == 1:
            if split_component[0].isnumeric():
                # component is just a number
                variable = 'null'
                coefficient = int(split_component[0])
            else:
                # component is just a variable (coefficient is 1)
                variable = split_component[0]
                coefficient = 1
        else:
            # component is coefficient*variable
            if split_component[0].isnumeric() and split_component[1].isalpha():
                variable = split_component[1]
                coefficient = int(split_component[0])
            # component is variable*coefficient
            else:
                variable = split_component[0]
                coefficient = int(split_component[1])
        if signs[i] == '-':
            coefficient = -coefficient
        if variable not in coefficients:
            coefficients[variable] = [coefficient]
        else:
            coefficients[variable].append(coefficient)
    # aggregate coefficients
    return {variable: sum(coefficients[variable]) for variable in coefficients}

def solve_linsys(system):
    '''
    solve a system of linear equations.
    
    :param system: either a string or a list of strings
    :return: dict mapping variable names to values
    '''
    def setup_linear_system(all_coefficients, ordered_variables):
        rows = list()
        for coefficients in all_coefficients:
            rows.append([coefficients.get(var, 0) for var in ordered_variables])
        target_vector = [coefficients.get('null', 0) for coefficients in all_coefficients]
        return np.array(rows), np.array(target_vector)

    if type(system) == str:
        system = [system]
    all_coefficients = list()
    variables = list()
    for linear_equation in system:
        coefficients = extract_coefficients(linear_equation)
        all_coefficients.append(coefficients)
        variables.extend(list(coefficients.keys()))
    ordered_variables = list(set(variables)-set(['null']))
    coefficient_matrix, target_vector = setup_linear_system(all_coefficients, ordered_variables)
    solution = np.linalg.solve(coefficient_matrix, target_vector)
    return {symbol: solution[i] for i, symbol in enumerate(ordered_variables)}

def add_keypair(ls, value):
    pass

# arithmetic


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