import os
import unittest

from gen_tsne.gen_tsne import calculate

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")


class TestGenTsne(unittest.TestCase):

    def test_build(self):
        paths = [os.path.join(ASSETS_DIR, "dataset"), os.path.join(ASSETS_DIR, "model_a"),
                 os.path.join(ASSETS_DIR, "model_b")]
        stats_df = calculate(paths, os.path.join(ASSETS_DIR, "output"), pca_components=None, frow=10, fcol=10)
        self.assertIn("jaccard_index", stats_df.columns)

    def test_build_with_features(self):
        paths = [os.path.join(ASSETS_DIR, "dataset"), os.path.join(ASSETS_DIR, "model_a")]
        stats_df = calculate(paths, os.path.join(ASSETS_DIR, "output"), pca_components=None, frow=10, fcol=10,
                             use_features=True)
        self.assertIn("jaccard_index", stats_df.columns)
