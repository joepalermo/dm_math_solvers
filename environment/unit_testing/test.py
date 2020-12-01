from environment.operators import extract_coefficients, solve_linsys

# extract_coefficients ---------------------------------

# signs
assert extract_coefficients('x=y') == {'x': 1, 'y': 1}
assert extract_coefficients('-x=y') == {'x': -1, 'y': 1}
assert extract_coefficients('x=-y') == {'x': 1, 'y': -1}
assert extract_coefficients('-x=-y') == {'x': -1, 'y': -1}

# spaces
assert extract_coefficients(' x = y ') == {'x': 1, 'y': 1}
assert extract_coefficients(' - x = y ') == {'x': -1, 'y': 1}
assert extract_coefficients(' x = - y ') == {'x': 1, 'y': -1}
assert extract_coefficients(' - x = - y ') == {'x': -1, 'y': -1}

# coefficients
assert extract_coefficients('2*x=y') == {'x': 2, 'y': 1}
assert extract_coefficients('-2*x=3*y') == {'x': -2, 'y': 3}
assert extract_coefficients('7*x=-3*y') == {'x': 7, 'y': -3}
assert extract_coefficients('-10*x=-11*y') == {'x': -10, 'y': -11}

# coefficients with null
assert extract_coefficients('2*x=y+5') == {'x': 2, 'y': 1, 'null': 5}
assert extract_coefficients('-2*x=3*y-1') == {'x': -2, 'y': 3, 'null': -1}
assert extract_coefficients('7*x=-3*y+11') == {'x': 7, 'y': -3, 'null': 11}
assert extract_coefficients('-10*x=-11*y-5') == {'x': -10, 'y': -11, 'null': -5}

# multiple of same coefficients
assert extract_coefficients('2*x + 1 = - x + y - 4') == {'x': 1, 'y': 1, 'null': -3}

# solve_linsys ---------------------------------

system = ['3*x + y = 9', 'x + 2*y = 8']
assert solve_linsys(system) == {'y': 3.0, 'x': 2.0}