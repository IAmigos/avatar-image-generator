import os
import pandas as pd
from sklearn.decomposition import PCA
from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def apply_tsne(df_feature_vector_info, feature_vectors, perplexity, n_iter, learning_rate=200, pca_components=None, tsne_jobs=4):
    if pca_components:
        print("shape before pca: %s", feature_vectors.shape)
        pca = PCA(n_components=pca_components, svd_solver='randomized')
        feature_vectors = pca.fit_transform(feature_vectors)
        pca_cols = [f"pca_{c}" for c in range(pca.n_components)]
        df_feature_vector_info[pca_cols] = pd.DataFrame(feature_vectors, index=df_feature_vector_info.index)
        print("shape after pca: %s", feature_vectors.shape)
    print();print('TSNE:')
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=n_iter, learning_rate=learning_rate,
                n_jobs=tsne_jobs)
    print()
    return tsne.fit_transform(feature_vectors), df_feature_vector_info

def generate_scatter(tsne_results, df_feature_vector_info, save_image, output_dir, save_wandb):
    if save_image:
        plt.figure(figsize=(10, 10))
        sns.scatterplot(x=tsne_results[:,0], y=tsne_results[:,1], hue=df_feature_vector_info['name'], legend="full", alpha=0.8)
        plt.savefig(os.path.join(output_dir, f"models_scatter.png"))
    if save_wandb:
        pass
