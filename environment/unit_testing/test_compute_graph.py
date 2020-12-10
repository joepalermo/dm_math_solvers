import unittest
from environment.utils import extract_formal_elements
from environment.typed_operators import lookup_value, solve_system, append, make_equality, lookup_value_eq, project_lhs, \
    substitution_left_to_right, extract_isolated_variable, factor, simplify, diff, replace_arg, make_function
from environment.compute_graph import ComputeGraph, Node


class Test(unittest.TestCase):

    def test_easy_algebra__linear_1d(self):
        problem_statement = 'Solve 0 = 4*b + b + 15 for b.'
        f = extract_formal_elements(problem_statement)
        cg = ComputeGraph(problem_statement)
        lookup_value_node = Node(lookup_value)
        solve_system_node = Node(solve_system)
        solve_system_node.set_arg(f[0])
        lookup_value_node.set_args([solve_system_node, f[1]])
        cg.root = lookup_value_node
        assert str(cg) == "lookup_value(solve_system('0 = 4*b + b + 15'),'b')"
        assert cg.eval() == -3

    def test_incomplete_compute_graph(self):
        problem_statement = 'Solve 0 = 4*b + b + 15 for b.'
        cg = ComputeGraph(problem_statement)
        lookup_value_node = Node(lookup_value)
        solve_system_node = Node(solve_system)
        lookup_value_node.set_arg(solve_system_node)
        cg.root = lookup_value_node
        assert str(cg) == "lookup_value(solve_system('param_0'),'param_1')"
        assert cg.eval() == None

