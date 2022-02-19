import os
import pandas as pd
from sklearn.decomposition import PCA
from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import io
import PIL

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

def generate_scatter(tsne_results, df_feature_vector_info, save_image, output_dir, save_wandb, plot_title):
        
    plt.figure(figsize=(10, 10))
    plt.title(plot_title)
    plt.xlabel('tsne_x')
    plt.ylabel('tsne_y')
    sns.scatterplot(x=tsne_results[:,0], y=tsne_results[:,1], hue=df_feature_vector_info['name'], legend="full", alpha=0.8)
    
    if save_image:
        plt.savefig(os.path.join(output_dir, plot_title+"_scatter_plot.png"))
                
    wandb_scatter_plot =None
    img_scatter_plot = None
    if save_wandb:
        data = [[x,y, name] for (x, y, name) in zip(list(tsne_results[:,0]), list(tsne_results[:,1]), list(df_feature_vector_info['name']))]
        table = wandb.Table(data=data, columns = ["tsne_x", "tsne_y",'name'])
        wandb_scatter_plot =  wandb.plot.scatter(table, "tsne_x", "tsne_y", title=plot_title)
#         wandb.log({"tsne evaluation" : wandb.plot.scatter(table, "tsne_x", "tsne_y", title="t-SNE evaluation")})
        
        buf = io.BytesIO()
        plt.savefig(buf)
        buf.seek(0)
        img_scatter_plot =  PIL.Image.open(buf)
        
    return wandb_scatter_plot, img_scatter_plot
