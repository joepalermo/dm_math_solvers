from inspect import signature


class Node:

    def __init__(self, action):
        self.action = action
        self.args = []
        if type(self.action) == 'str':  # if action is a formal element
            self.num_parameters = 0
        else:
            self.num_parameters = len(signature(self.action).parameters)

    def set_arg(self, node):
        assert len(self.args) < self.num_parameters
        self.args.append(node)

    def args_set(self):
        return len(self.args) == self.num_parameters


class ComputeGraph:

    def __init__(self):
        self.root = None
        self.current_node = None  # reference to the first node (breadth-first) that requires one or more arguments
        self.queue = []

    def __str__(self):
        '''
        traverse the graph breath-first to construct a string representing the compute graph.
        :return:
        '''
        pass

    def eval(self):
        '''
        evaluate the compute graph
        :return: the output of the compute graph
        '''
        pass

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
            new_node = Node(action)
            self.current_node.set_arg(Node(action))
            self.queue.append(new_node)  # add new node to queue for later processing
            if self.current_node.args_set():
                self.current_node = self.queue.pop()

    def reset(self):
        self.root = None
        self.current_node = None
        self.queue = []