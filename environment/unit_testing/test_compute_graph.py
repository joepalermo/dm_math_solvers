import unittest
from environment.utils import extract_formal_elements
from environment.typed_operators import *
from environment.compute_graph import ComputeGraph, Node


class Test(unittest.TestCase):
    def test_easy_algebra__linear_1d(self):
        problem_statement = "Solve 0 = 4*b + b + 15 for b."
        f = extract_formal_elements(problem_statement)
        cg = ComputeGraph(problem_statement)
        lookup_value_node = Node(lookup_value)
        solve_system_node = Node(solve_system)
        append_to_empty_list_node = Node(append_to_empty_list)
        append_to_empty_list_node.set_arg(Node('f0'))
        solve_system_node.set_arg(append_to_empty_list_node)
        lookup_value_node.set_args([solve_system_node, Node('f1')])
        cg.root = lookup_value_node
        assert str(cg) == "lookup_value(solve_system(append_to_empty_list(Equation('0 = 4*b + b + 15'))),Variable('b'))"
        assert cg.eval() == Value(-3)

    def test_incomplete_compute_graph(self):
        problem_statement = "Solve 0 = 4*b + b + 15 for b."
        cg = ComputeGraph(problem_statement)
        lookup_value_node = Node(lookup_value)
        solve_system_node = Node(solve_system)
        lookup_value_node.set_arg(solve_system_node)
        cg.root = lookup_value_node
        assert str(cg) == "lookup_value(solve_system('param_0'),'param_1')"
        assert cg.eval() == None
