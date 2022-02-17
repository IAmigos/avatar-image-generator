import argparse
import logging
import os

from tsne_evaluation_utils.grid import build as build_grid
from tsne_evaluation_utils.metric import calc_jaccard_index, calc_rmse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Code as in https://github.com/vfcosta/gen-tsne

def calculate(paths, output_dir, enable_rmse=True, pca_components=None, frow=60, fcol=60, perplexity=30,
              n_iter=1000, save_data=True, use_features=False):
    df, image_shape = build_grid(paths, pca_components=pca_components, frow=frow, fcol=fcol, perplexity=perplexity,
                                 n_iter=n_iter, save_data=save_data, output_dir=output_dir, use_features=use_features)
    distance_threshold, stats_df = calc_jaccard_index(df)
    logger.info("stats %s", stats_df)
    stats_df.to_csv(os.path.join(output_dir, "stats.csv"), index=False)
    if enable_rmse:
        df_distances = calc_rmse(df, image_shape)
        df_distances.to_csv(os.path.join(output_dir, "distances.csv"), index=False)
    return stats_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply Gen t-SNE metric.')
    parser.add_argument('-b', '--baseline', help='Path to images from the dataset (baseline)', required=True)
    parser.add_argument('-p', '--paths', action='append', help='Paths to images from generative models', required=True)
    parser.add_argument('-o', '--output', help='Output dir', default="./output")
    parser.add_argument('-r', "--rows", type=int, help='rows', default=60)
    parser.add_argument('-c', "--cols", type=int, help='cols', default=60)
    parser.add_argument('-k', "--perplexity", type=int, help='perplexity', default=30)
    parser.add_argument('-n', "--iter", type=int, help='iterations', default=1000)
    parser.add_argument('-f', "--use-features", default=False, action='store_true',
                        help='Use features to build the grid (<IMAGE_NAME>.npy or <IMAGE_NAME>.npz)')
    args = parser.parse_args()
    calculate([args.baseline] + args.paths, args.output, frow=args.rows, fcol=args.cols, perplexity=args.perplexity,
              n_iter=args.iter, use_features=args.use_features)
