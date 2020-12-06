from inspect import signature
from environment.operators import append, add_keypair, lookup_value, function_application, apply_mapping, calc, \
    make_equality, project_lhs, project_rhs, simplify, solve_system, factor, diff, replace_arg, substitution_left_to_right, \
    eval_in_base, root, round_to_int, round_to_dec, power, substitution_right_to_left, max_arg, min_arg, greater_than, \
    less_than, lookup_value_eq


class Node:

    def __init__(self, action):
        self.action = action
        self.args = []
        if type(self.action) == str:  # if action is a formal element
            self.num_parameters = 0
        else:
            self.num_parameters = len(signature(self.action).parameters)

    def set_arg(self, node):
        assert len(self.args) < self.num_parameters
        self.args.append(node)

    def set_args(self, nodes):
        assert len(self.args) == 0
        self.args = nodes

    def args_set(self):
        return len(self.args) == self.num_parameters


def build_string(current_node):
    if type(current_node) == str:
        return f"'{current_node}'"
    elif type(current_node.action) == str:
        return f"'{current_node.action}'"
    elif current_node.action is None:
        return 'None'
    else:
        arg_strings = []
        if len(current_node.args) < current_node.num_parameters:
            num_params = current_node.num_parameters
            num_args = len(current_node.args)
            args = current_node.args + [f"param_{i}" for i in range(num_args, num_params)]
        else:
            args = current_node.args
        for arg in args:
            arg_string = build_string(arg)
            arg_strings.append(arg_string)
        return f"{current_node.action.__name__}({','.join(['{}'.format(arg_string) for arg_string in arg_strings])})"


class ComputeGraph:

    def __init__(self):
        self.root = None
        self.current_node = None  # reference to the first node (breadth-first) that requires one or more arguments
        self.queue = []

    def __str__(self):
        '''
        traverse the graph to construct a string representing the compute graph.
        :return:
        '''
        return build_string(self.root)

    def eval(self):
        '''
        evaluate the compute graph
        :return: the output of the compute graph
        '''
        try:
            return eval(str(self))
        except:
            return None

    def add(self, action):
        '''
        Add an action to the compute graph. Elements are added breadth-first order.

        :param action: either an operator or a formal element
        '''
        if self.root is None:
            self.root = Node(action)
            if not self.root.args_set():
                self.current_node = self.root
            else:
                self.current_node = None
        else:
            new_node = Node(action)
            self.current_node.set_arg(Node(action))
            if new_node.num_parameters > 0:
                self.queue.append(new_node)  # add new node to queue for later processing
            if self.current_node.args_set():
                if len(self.queue) > 0:
                    self.current_node = self.queue.pop()
                else:
                    self.current_node = None

    def reset(self):
        self.root = None
        self.current_node = None
        self.queue = []
