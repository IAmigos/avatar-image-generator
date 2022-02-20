import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .utils_tsne import apply_tsne, generate_scatter

def tsne_evaluation(ls_feature_arrays, ls_array_names, pca_components=None, perplexity=30, n_iter=1000, save_image=False, output_dir='./', save_wandb=False, plot_title='t-SNE evaluation'):
    assert len(ls_feature_arrays) == len(ls_array_names)
    
    feature_vectors = np.concatenate(ls_feature_arrays)

    # cancatenate names in a df with same length as feature_vectors
    feature_vector_names = []
    list( map(feature_vector_names.extend, [[name]*ls_feature_arrays[i].shape[0] for i, name in enumerate(ls_array_names)]) )     
    df_feature_vector_info = pd.DataFrame({'name':feature_vector_names})
    
    tsne_results, df_feature_vector_info = apply_tsne(df_feature_vector_info , feature_vectors, perplexity, n_iter, pca_components=pca_components)
    
    tsne_results_norm =  StandardScaler().fit_transform(tsne_results)
    
    scatter_plot = None
    wandb_scatter_plot = None
    img_scatter_plot = None
    if save_image or save_wandb:
        wandb_scatter_plot, img_scatter_plot = generate_scatter(tsne_results_norm, df_feature_vector_info, save_image, output_dir, save_wandb, plot_title)
        
   
    return tsne_results_norm, df_feature_vector_info, wandb_scatter_plot, img_scatter_plot

    ###############################
    
#     distance_threshold, stats_df = calc_jaccard_index(df)
#     logger.info("stats %s", stats_df)
#     stats_df.to_csv(os.path.join(output_dir, "stats.csv"), index=False)
#     if enable_rmse:
#         df_distances = calc_rmse(df, image_shape)
#         df_distances.to_csv(os.path.join(output_dir, "distances.csv"), index=False)
#     return stats_df


if __name__ == "__main__":
    output_dir = "./"
#     pca_components = 10
    pca_components = None
    with open('dataset.npy', 'rb') as f:
        np_a = np.load(f)
    with open('model_a.npy', 'rb') as f:
        np_b = np.load(f)
    with open('model_b.npy', 'rb') as f:
        np_c = np.load(f)
        
    tsne_evaluation([np_a, np_b, np_c],['dataset', 'm_a','m_b'],  pca_components=pca_components, save_image=True, output_dir= output_dir, save_wandb = True)
