"""Unit tests for predefined configurations."""

import unittest
from linkscluster import configs
from linkscluster import links_clusterer

LinksClusterer = links_clusterer.LinksClusterer


class TestConfigs(unittest.TestCase):
  """Tests for configs module."""

  def test_default_config(self):
    clusterer = configs.default_links_clusterer
    self.assertIsInstance(clusterer, LinksClusterer)
    self.assertEqual(clusterer.tc, 0.5)
    self.assertEqual(clusterer.ts, 0.8)
    self.assertEqual(clusterer.tp, 1.0)

  def test_facenet_config(self):
    clusterer = configs.facenet_clusterer
    self.assertIsInstance(clusterer, LinksClusterer)
    self.assertEqual(clusterer.tc, 0.6)
    self.assertEqual(clusterer.ts, 0.85)
    self.assertEqual(clusterer.tp, 0.95)

  def test_ge2e_voice_config(self):
    clusterer = configs.ge2e_voice_clusterer
    self.assertIsInstance(clusterer, LinksClusterer)
    self.assertEqual(clusterer.tc, 0.55)
    self.assertEqual(clusterer.ts, 0.8)
    self.assertEqual(clusterer.tp, 0.9)


if __name__ == "__main__":
  unittest.main()
