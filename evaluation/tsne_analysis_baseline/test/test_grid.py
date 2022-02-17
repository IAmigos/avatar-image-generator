import os
import unittest

from gen_tsne import build_grid

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")


class TestGrid(unittest.TestCase):

    def test_build(self):
        paths = [os.path.join(ASSETS_DIR, "dataset"), os.path.join(ASSETS_DIR, "model_a"),
                 os.path.join(ASSETS_DIR, "model_b")]
        df, _ = build_grid(paths, output_dir=os.path.join(ASSETS_DIR, "output"), pca_components=None, frow=10, fcol=10,
                           save_scatter=True)
        self.assertIn("tsne_x", df.columns)
        self.assertIn("tsne_y", df.columns)

    def test_build_pca(self):
        paths = [os.path.join(ASSETS_DIR, "dataset"), os.path.join(ASSETS_DIR, "model_a"),
                 os.path.join(ASSETS_DIR, "model_b")]
        df, _ = build_grid(paths, output_dir=os.path.join(ASSETS_DIR, "output"), pca_components=5, frow=10, fcol=10)
        self.assertIn("tsne_x", df.columns)
        self.assertIn("tsne_y", df.columns)

    def test_build_with_features(self):
        paths = [os.path.join(ASSETS_DIR, "dataset"), os.path.join(ASSETS_DIR, "model_a"),
                 os.path.join(ASSETS_DIR, "model_b")]
        df, _ = build_grid(paths, output_dir=os.path.join(ASSETS_DIR, "output"), pca_components=None, frow=10, fcol=10,
                           use_features=True)
        self.assertIn("tsne_x", df.columns)
        self.assertIn("tsne_y", df.columns)
