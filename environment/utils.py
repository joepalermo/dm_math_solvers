import re


def is_numeric(string):
    return all([x.isnumeric() or x == '.' for x in string] + [string.count('.') <= 1])


def extract_formal_elements(problem_statement):
    pattern = '\$f\[(.+?)\]'
    return re.findall(pattern, problem_statement)
