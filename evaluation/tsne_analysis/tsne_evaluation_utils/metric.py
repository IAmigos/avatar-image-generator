import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)


def calc_jaccard_index(df):
    df_dataset = df[df["name"] == df.iloc[0]["name"]]
    df_models = df[df["name"] != df.iloc[0]["name"]].reset_index()
    distance_matrix = _calc_distances(df_dataset, df_models)
    min_distances_matrix = distance_matrix.min(axis=1)
    logger.info("distance matrix percentile %f", np.percentile(min_distances_matrix, 50))
    distance_threshold = np.percentile(min_distances_matrix, 50)
    logger.info("distance_threshold: %f", distance_threshold)
    model_names = df_models["name"].unique()
    cols = ["selected", "distance_threshold", "intersection", "jaccard_index"]
    stats_df = pd.DataFrame(index=model_names, columns=cols)
    for name in model_names:
        df_model = df_models[df_models["name"] == name]
        all_selected = set()
        min_distances, intersection_gen = [], []
        for i, row in df_model.iterrows():
            distances = distance_matrix[i]
            selected = np.where(distances < distance_threshold)[0]
            min_distances.append(np.min(distances))
            if len(selected) > 0:
                all_selected = all_selected.union(selected)
                intersection_gen.append(i)
        logger.info("model %s selected: %d intersection: %d", name, len(all_selected), len(intersection_gen))
        stats_df.loc[name]["selected"] = len(all_selected)
        stats_df.loc[name]["intersection"] = len(intersection_gen)
        stats_df.loc[name]["jaccard_index"] = len(intersection_gen)/(len(df_model) + len(df_dataset) - len(intersection_gen))
    stats_df["distance_threshold"] = distance_threshold
    return distance_threshold, stats_df.reset_index()


def _calc_distances(df_dataset, df_models):
    distance_matrix = np.empty((len(df_models), len(df_dataset)))
    for i, row in df_models.iterrows():
        distances = np.sqrt(np.sum((df_dataset[["tsne_x", "tsne_y"]] - row[["tsne_x", "tsne_y"]]) ** 2, axis=1))
        distance_matrix[i] = distances
    return distance_matrix


def calc_rmse(df, shape):
    dataset_name = df.iloc[0]["name"]
    df_dataset = df[df["name"] == dataset_name]
    df_models = df[df["name"] != dataset_name]
    row_distances = []
    for _, row in df_models.iterrows():
        distances = np.sqrt(np.sum((df_dataset[["tsne_x", "tsne_y"]] - row[["tsne_x", "tsne_y"]])**2, axis=1))
        min_index = np.argmin(distances)
        max_index = np.argmax(distances)
        cols = list(range(np.prod(shape)))
        cols_act = [c for c in df_dataset.columns if str(c).startswith("act_")]
        cols_pca = [c for c in df_dataset.columns if str(c).startswith("pca_")]

        values = {}
        for k, index in {"min": min_index, "max": max_index}.items():
            row_dataset = df_dataset.iloc[index]
            rmse = np.sqrt(mean_squared_error(row[cols], row_dataset[cols]))
            rmse_act = np.sqrt(mean_squared_error(row[cols_act], row_dataset[cols_act])) if cols_act else None
            rmse_pca = np.sqrt(mean_squared_error(row[cols_pca], row_dataset[cols_pca])) if cols_pca else None
            values = {**values, f"rmse_{k}": rmse, f"rmse_act_{k}": rmse_act, f"rmse_pca_{k}": rmse_pca,
                      f"distance_{k}": distances[index], f"index_{k}": index}
        row_distances.append(values)
    return pd.DataFrame(row_distances)
