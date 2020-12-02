import re
import numpy as np
from environment.utils import is_numeric


def extract_coefficients(linear_equation):
    '''
    :param linear_equation:
    :return: dict mapping symbol to coefficient
    '''
    # component pattern matches a single component of a linear equation (e.g. -1 or 3*x or -t*2, etc...)
    equality_index = linear_equation.index('=')
    component_pattern = '[\+|-]?[.a-zA-Z0-9\s]*\*?[.a-zA-Z0-9\s]*'
    components = list()
    # iterate through all non-empty matches
    for match in [m for m in re.finditer(component_pattern, linear_equation) if len(m.group(0).strip()) > 0]:
        component = dict()
        component['term'] = match.group(0).strip()
        component['side'] = 'lhs' if match.start(0) < equality_index else 'rhs'
        # identify the sign
        if component['term'][0] == '-':
            component['sign'] = -1
            component['term_without_sign'] = component['term'][1:].strip()
        elif component['term'][0] == '+':
            component['sign'] = 1
            component['term_without_sign'] = component['term'][1:].strip()
        else:
            component['sign'] = 1
            component['term_without_sign'] = component['term']
        # identify the coefficient and the variable (apply sign)
        if '*' in component['term_without_sign']:
            t1,t2 = component['term_without_sign'].split('*')
            if is_numeric(t1):
                component['coefficient'] = component['sign'] * float(t1)
                component['variable'] = t2
            else:
                component['coefficient'] = component['sign'] * float(t2)
                component['variable'] = t1
        else:
            if is_numeric(component['term_without_sign']):
                component['coefficient'] = component['sign'] * float(component['term_without_sign'])
                component['variable'] = None
            else:
                component['coefficient'] = component['sign'] * 1
                component['variable'] = component['term_without_sign']
        components.append(component)
    coefficients = dict()
    for component in components:
        # flip sign if appropriate (based on side)
        if (component['side'] == 'rhs' and component['variable'] is not None) or \
           (component['side'] == 'lhs' and component['variable'] is None):
            component['side_adjusted_coefficient'] = -component['coefficient']
        else:
            component['side_adjusted_coefficient'] = component['coefficient']
        # build a dictionary that maps variables to a list of coefficients to be aggregated
        if component['variable'] not in coefficients:
            coefficients[component['variable']] = [component['side_adjusted_coefficient']]

        else:
            coefficients[component['variable']].append(component['side_adjusted_coefficient'])
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
        target_vector = [coefficients.get(None, 0) for coefficients in all_coefficients]
        return np.array(rows), np.array(target_vector)

    if type(system) == str:
        system = [system]
    all_coefficients = list()
    variables = list()
    for linear_equation in system:
        coefficients = extract_coefficients(linear_equation)
        all_coefficients.append(coefficients)
        variables.extend(list(coefficients.keys()))
    ordered_variables = list(set(variables) - set([None]))
    coefficient_matrix, target_vector = setup_linear_system(all_coefficients, ordered_variables)
    solution = np.linalg.solve(coefficient_matrix, target_vector)
    return {symbol: solution[i] for i, symbol in enumerate(ordered_variables)}
