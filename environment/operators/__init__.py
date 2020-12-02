def append(ls, value):
    if not ls:
        return [value]
    else:
        ls.append(value)
        return ls


def add_key_pair(mapping, key, value):
    if not mapping:
        return {key: value}
    else:
        mapping[key] = value
        return mapping


def lookup_value(mapping, key):
    return mapping.get(key, None)


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