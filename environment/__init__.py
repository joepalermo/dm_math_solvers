import re


def extract_formal_elements(problem_statement):
    pattern = '\$f\[(.+?)\]'
    return re.findall(pattern, problem_statement)
