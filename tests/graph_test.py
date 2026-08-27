"""Unit tests for SubclusterGraph class."""

import unittest
from linkscluster import graph


class TestSubclusterGraph(unittest.TestCase):
  """Tests for SubclusterGraph class."""

  def test_nodes_and_edges(self):
    g = graph.SubclusterGraph()
    g.add_node(0)
    g.add_node(1)
    g.add_edge(0, 1)

    self.assertTrue(g.has_edge(0, 1))
    self.assertTrue(g.has_edge(1, 0))
    self.assertIn(1, g.neighbors(0))
    self.assertEqual(g.degree(0), 1)

    g.remove_edge(0, 1)
    self.assertFalse(g.has_edge(0, 1))
    self.assertEqual(g.degree(0), 0)

  def test_merge_nodes(self):
    g = graph.SubclusterGraph()
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(1, 3)

    # Merge node 1 into node 0
    g.merge_nodes(keep_id=0, remove_id=1)

    self.assertFalse(g.has_edge(0, 1))
    self.assertTrue(g.has_edge(0, 2))
    self.assertTrue(g.has_edge(0, 3))
    self.assertNotIn(1, g.adj)

  def test_check_severed(self):
    g = graph.SubclusterGraph()
    g.add_edge(0, 1)
    g.add_edge(1, 2)

    # 0 and 2 connected via 1
    severed, comp = g.check_severed(start_id=0, target_id=2)
    self.assertFalse(severed)
    self.assertEqual(comp, set())

    # Disconnect
    g.remove_edge(1, 2)
    severed, comp = g.check_severed(start_id=2, target_id=0)
    self.assertTrue(severed)
    self.assertEqual(comp, {2})

  def test_cycle_not_severed(self):
    g = graph.SubclusterGraph()
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 0)

    # Remove (0, 1); 0 and 1 still connected via 2
    g.remove_edge(0, 1)
    severed, comp = g.check_severed(start_id=1, target_id=0)
    self.assertFalse(severed)

  def test_get_component(self):
    g = graph.SubclusterGraph()
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_node(3)

    comp = g.get_component(0)
    self.assertEqual(comp, {0, 1, 2})


if __name__ == "__main__":
  unittest.main()
