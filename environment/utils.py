import re
from environment.typed_operators import Eq, Fn, Ex, Var, Val


def is_numeric(string):
    return all([x.isnumeric() or x == "." for x in string] + [string.count(".") <= 1])


def extract_formal_elements_as_annotations(problem_statement):
    pattern = "\$f\[(.+?)\]"
    return re.findall(pattern, problem_statement)


def extract_formal_elements(question, cast=True):
    # split on punctuation unless it is immediately preceded and followed by a number (indicating it is a decimal)
    split_on_punctuation = "***".join(
        [
            string
            for string in re.split("(?<![0-9])[.,;:?]|[.,;:?](?![0-9])", question)
            if len(string) > 0 and not string.isspace()
        ]
    )
    # TODO: use a more sophisticated mechanism (CFG?) to math expressions, equations, etc... this could account for variables names that have length greater than 1
    split_on_words = [
        string
        for string in re.split("[A-Za-z]\w+|\*\*\*", split_on_punctuation)
        if len(string) > 0 and not string.isspace()
    ]
    # strip trailing or leading whitespace
    formal_elements = [string.strip() for string in split_on_words]
    # filter for the special case where the letter "a" gets included at the end of a formal element
    formal_elements = [
        f if len(re.findall("[0-9A-Za-z\)](\sa)", f)) < 1 else f.split(" a")[0]
        for f in formal_elements
    ]
    # cast types
    if cast:
        formal_elements = [cast_formal_element(f) for f in formal_elements]
    return formal_elements


def cast_formal_element(f):
    if "=" in f:
        try:
            return Fn(f)
        except:
            return Eq(f)
    elif len(f) == 1 and f.isalpha():
        return Var(f)
    elif f.isnumeric():
        return Val(f)
    else:
        return Ex(f)


def guess_until_problem_solved(
    env, problem_index, verbose=False, max_episode_index=1000
):
    episode_i = 0
    graph_guessed_correctly = False
    encoded_problem_statement, _ = env.reset_with_specific_problem(
        "short_problems", 0, problem_index
    )
    print(f"\nproblem statement: {env.decode(encoded_problem_statement)}")
    while not graph_guessed_correctly and episode_i < max_episode_index:
        _, _ = env.reset_with_specific_problem("short_problems", 0, problem_index)
        done = False
        step_i = 0
        if verbose:
            print(f"episode: {episode_i}")
        while not done:
            action_index = env.sample_masked_action_index()
            observation, reward, done, info = env.step(action_index)
            # if verbose:
            #     print(f"\t\tS': {observation}, R: {reward}, done: {done}")
            if reward == 1:
                graph_guessed_correctly = True
            step_i += 1
        episode_i += 1
    print(f'graph: {info["raw_observation"].split(";")[1]}')
    print(f"trials taken to guess problem #{problem_index}: {episode_i}")


def filter_univariate(examples):
    univariate_examples = []
    for question, answer in examples:
        formal_elements = extract_formal_elements(question, cast=False)
        function = formal_elements[0]
        num_vars = len([ch for ch in set(function) if ch.isalpha()])
        if num_vars == 1:
            univariate_examples.append((question, answer))
    return univariate_examples
