import unittest
from environment.utils import extract_formal_elements
from environment.typed_operators import *
from environment.compute_graph import ComputeGraph, Node


class Test(unittest.TestCase):
    def test_easy_algebra__linear_1d(self):
        question = "Solve 0 = 4*b + b + 15 for b."
        f = extract_formal_elements(question)
        cg = ComputeGraph(question)
        lookup_value_node = Node(lv)
        solve_system_node = Node(ss)
        append_to_empty_list_node = Node(ape)
        append_to_empty_list_node.set_arg(Node('f0'))
        solve_system_node.set_arg(append_to_empty_list_node)
        lookup_value_node.set_args([solve_system_node, Node('f1')])
        cg.root = lookup_value_node
        assert str(cg) == "lv(ss(ape(Eq('0 = 4*b + b + 15'))),Var('b'))"
        assert cg.eval() == Val(-3)

    def test_incomplete_compute_graph(self):
        question = "Solve 0 = 4*b + b + 15 for b."
        cg = ComputeGraph(question)
        lookup_value_node = Node(lv)
        solve_system_node = Node(ss)
        lookup_value_node.set_arg(solve_system_node)
        cg.root = lookup_value_node
        assert str(cg) == "lv(ss('p_0'),'p_1')"
        assert cg.eval() == None
