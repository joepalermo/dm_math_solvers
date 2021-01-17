import re
import tqdm as tqdm
from environment.typed_operators import Eq, Fn, Ex, Var, Val, Rat


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
    elif re.compile("([0-9]+[/][0-9]+$)").match(f):
        return Rat(f)
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


def get_module_name_from_filepath(fp):
    module_name = fp.split("/")[-1].split(".txt")[0]
    if "compose" in module_name:
        module_name = module_name.split("_compose")[0]
    else:
        module_name = module_name
    return module_name


# load train data
def load_training_data(config):
    train = {}
    print("loading problems")
    for filepath in tqdm(config.problem_filepaths):
        with open(filepath, "r") as f:
            lines = f.readlines()
        num_pairs = min(len(lines) // 2, config.num_problems_per_module)
        for i in range(0, 2 * num_pairs, 2):
            question = lines[i].strip()
            answer = lines[i + 1].strip()
            # for uncomposed problems set difficulty to 0 to distinguish them
            difficulty = (
                len(re.split("(?<![0-9])[.,;:?]|[.,;:?](?![0-9])", question)) - 1
                if 'compose' in filepath
                else 0
            )
            # don't load problems with difficulty above the maximum
            if difficulty > config.max_difficulty:
                continue
            module_name = get_module_name_from_filepath(filepath)
            if module_name in train:
                if difficulty in train[module_name]:
                    train[module_name][difficulty].append((question, answer))
                else:
                    train[module_name][difficulty] = [(question, answer)]
            else:
                train[module_name] = {difficulty: [(question, answer)]}
    if config.univariate_differentiation:
        train['calculus__differentiate'][0] = filter_univariate(train['calculus__differentiate'][0])
    return train


def split_validation_data(config, train):
    val = {}
    for module_name in train:
        val[module_name] = {}
        for difficulty in train[module_name]:
            num_examples = len(train[module_name][difficulty])
            num_val = int(num_examples * config.validation_percentage)
            val[module_name][difficulty] = train[module_name][difficulty][:num_val]
            train[module_name][difficulty] = train[module_name][difficulty][num_val:]
            assert (
                len(train[module_name][difficulty])
                + len(val[module_name][difficulty])
                == num_examples
            )
    return val

def build_tokenizer(config)
    self.padding_token = config.vocab_size
    self.tokenizer = Tokenizer(BPE())
    trainer = BpeTrainer(vocab_size=self.vocab_size, special_tokens=self.special_tokens)
    self.tokenizer.train(trainer, [str(Path(config.corpus_filepath).resolve())])