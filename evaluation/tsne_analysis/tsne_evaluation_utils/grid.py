import logging
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from MulticoreTSNE import MulticoreTSNE as TSNE
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def build(paths, frow=60, fcol=60, perplexity=30, n_iter=1000, jitter_win=0, pca_components=50,
          output_dir="./output", save_data=True, save_scatter=True, use_features=False):
    os.makedirs(output_dir, exist_ok=True)
    df, image_shape, tsne_input = load_data(paths, use_features)
    tsne_results = apply_tsne(df, tsne_input, perplexity, n_iter, pca_components=pca_components)
    logger.info("tsne finished: %s", tsne_results.shape)
    df['tsne_x_raw'], df['tsne_y_raw'] = tsne_results[:, 0], tsne_results[:, 1]
    norm = StandardScaler().fit_transform(df[["tsne_x_raw", "tsne_y_raw"]])
    df['tsne_x'], df['tsne_y'] = norm[:, 0], norm[:, 1]
    if save_scatter:
        generate_scatter(df, output_dir)
    df = generate_images(fcol, frow, image_shape, df, output_dir=output_dir, jitter_win=jitter_win)
    if save_data:
        logger.info("saving data.csv")
        df.to_csv(os.path.join(output_dir, "data.csv"), index=False)
    logger.info("finished")
    return df, image_shape


def generate_images(fcol, frow, image_shape, df, output_dir=None, jitter_win=None):
    df["tsne_x_int"] = ((fcol - 1) * (df["tsne_x"] - np.min(df["tsne_x"])) / np.ptp(df["tsne_x"])).astype(int)
    df["tsne_y_int"] = ((frow - 1) * (df["tsne_y"] - np.min(df["tsne_y"])) / np.ptp(df["tsne_y"])).astype(int)
    all_possibilities = []
    if jitter_win:
        yy, xx = np.mgrid[-jitter_win:jitter_win + 1, -jitter_win:jitter_win + 1]
        all_possibilities = np.vstack([xx.reshape(-1), yy.reshape(-1)]).T.tolist()
        all_possibilities.sort(key=lambda x: (max(abs(x[0]), abs(x[1])), abs(x[0]) + abs(x[1])))
        all_possibilities.pop(0)

    for model_name, group in df.groupby(by="name"):
        ordered_images = np.zeros((frow, fcol, *image_shape))
        overlap, show = 0, 0
        for i, row in group.iterrows():
            x, y = row["tsne_x_int"], row["tsne_y_int"]
            possibilities = list(all_possibilities)
            while len(possibilities) and np.sum(ordered_images[x, y]) != 0:
                dx, dy = possibilities.pop(0)
                x, y = np.clip(x + dx, 0, fcol - 1), np.clip(y + dy, 0, frow - 1)
            if np.sum(ordered_images[x, y]) == 0:
                show += 1
                ordered_images[x, y] = row[get_features(image_shape)].values.reshape((-1, *image_shape))
            else:
                overlap += 1
        logger.info("overlap for %s: %d, show: %d", model_name, overlap, show)
        ordered_images = np.flipud(np.transpose(ordered_images, (1, 0, 2, 3, 4))).reshape(frow * fcol, *image_shape)

        grid = (ordered_images.reshape(frow, fcol, *image_shape).swapaxes(1, 2)
                .reshape(image_shape[0] * frow, image_shape[1] * fcol, image_shape[2]))
        logger.info("tsne grid shape: %s", grid.shape)
        plt.figure(figsize=(20, 20))
        plt.imsave(os.path.join(output_dir, f"tsne_{model_name}.png"), grid)
    return df


def apply_tsne(df, data, perplexity, n_iter, learning_rate=200, pca_components=None, tsne_jobs=4):
    if pca_components:
        logger.info("shape before pca: %s", data.shape)
        pca = PCA(n_components=pca_components, svd_solver='randomized')
        data = pca.fit_transform(data)
        pca_cols = [f"pca_{c}" for c in range(pca.n_components)]
        df[pca_cols] = pd.DataFrame(data, index=df.index)
        logger.info("shape after pca: %s", data.shape)
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=n_iter, learning_rate=learning_rate,
                n_jobs=tsne_jobs)
    return tsne.fit_transform(data)


def load_features(image_path, extensions=("npz", "npy")):
    base_path = os.path.splitext(image_path)[0]
    for ext in extensions:
        f = f"{base_path}.{ext}"
        if os.path.exists(f):
            data = np.load(f)
            if ext == "npz":
                data = data["arr_0"]
            return data
    return None


def load_data(paths, use_features):
    df = pd.DataFrame()
    image_shape = None
    all_features = []
    for path in paths:
        logger.info("loading images from %s", path)
        name = os.path.basename(path)
        for f in glob(os.path.join(path, "*.png")):
            if use_features:
                features = load_features(f)
                if features is None:
                    logger.warning("features not found for %s", f)
                    continue
                all_features.append(features)
            image = np.array(Image.open(f))/255
            image_shape = image.shape
            df_new = pd.DataFrame(image.reshape((-1, np.prod(image_shape))))
            df_new["name"] = name
            df_new["file"] = f
            df = df.append(df_new)
    logger.info("loaded %d images with shape %s", len(df), image_shape)
    tsne_input = np.array(all_features) if use_features else get_image_data(df, image_shape)
    return df.reset_index(), image_shape, tsne_input


def generate_scatter(df, output_dir):
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x="tsne_x", y="tsne_y", hue="name", data=df, legend="full", alpha=0.2)
    plt.savefig(os.path.join(output_dir, f"models_scatter.png"))


def get_features(image_shape):
    return list(range(np.prod(image_shape)))


def get_image_data(df, image_shape):
    return df[get_features(image_shape)].values
