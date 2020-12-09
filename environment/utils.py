import re


def is_numeric(string):
    return all([x.isnumeric() or x == '.' for x in string] + [string.count('.') <= 1])


def extract_formal_elements_as_annotations(problem_statement):
    pattern = '\$f\[(.+?)\]'
    return re.findall(pattern, problem_statement)


def extract_formal_elements(question):
    # split on punctuation unless it is immediately preceded and followed by a number (indicating it is a decimal)
    split_on_punctuation = "***".join([string for string in re.split('(?<![0-9])[.,;:?]|[.,;:?](?![0-9])', question)
                                       if len(string) > 0 and not string.isspace()])
    # TODO: use a more sophisticated mechanism (CFG?) to math expressions, equations, etc... this could account for variables names that have length greater than 1
    split_on_words = [string for string in re.split('[A-Za-z]\w+|\*\*\*', split_on_punctuation)
                      if len(string) > 0 and not string.isspace()]
    # strip trailing or leading whitespace
    formal_elements = [string.strip() for string in split_on_words]
    # filter for the special case where the letter "a" gets included at the end of a formal element
    formal_elements = [f if len(re.findall('[0-9A-Za-z\)](\sa)', f)) < 1 else f.split(' a')[0] for f in formal_elements]
    return formal_elements