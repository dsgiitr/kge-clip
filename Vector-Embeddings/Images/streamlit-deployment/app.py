import numpy as np 
import pandas as pd 
import streamlit as st 
import torch 
from PIL import Image
from io import BytesIO
import requests
from sklearn.decomposition import PCA
import umap
from sklearn.manifold import TSNE, trustworthiness, Isomap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from transformers import CLIPProcessor, CLIPModel
from PIL import UnidentifiedImageError
import plotly.express as px
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from datasets import load_dataset

st.set_page_config(page_title="CLIP Image Embeddings", layout="wide")

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

model, processor = load_model()

@st.cache_data
def load_dataset_safely():
    try:
        dataset = load_dataset("recastai/coyo-75k-augmented-captions")
        return dataset['test']  # or 'train' if you prefer
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

test_dataset = load_dataset_safely()

def plot_images_grid(images, text, grid_size=(5, 5), titles=None, size=(5, 5), cmap=None):
    from math import ceil
    
    nrows, ncols = grid_size
    fig, axes = plt.subplots(nrows, ncols, figsize=size)
    
    if nrows == 1 and ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (img, txt) in enumerate(zip(images, text)):
        if isinstance(img, Image.Image):
            img = img.convert("RGB")
        if cmap:
            axes[idx].imshow(img, cmap=cmap)
        else:
            axes[idx].imshow(img)
        
        axes[idx].set_title(txt, fontsize=10)
        axes[idx].axis('off')
        
    for ax in axes[len(images):]:
        ax.axis('off')
    
    plt.tight_layout()
    st.sidebar.pyplot(fig)
@st.cache_data
def get_image_embeddings(data):
    error_count = 0
    error_urls = []
    image_embeddings = [] 
    c = 0
    
    for item in tqdm(data[:max_images]):
        image_url = item['url']
        text = item['llm_caption'][0]  # selecting 1st caption for image from list
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            if c < 5:
                plot_images_grid([image], [text], grid_size=(1, 1))
                c += 1

            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                image_embedding = model.get_image_features(**inputs)
            image_embeddings.append(image_embedding.squeeze().cpu().numpy())

        except Exception as e:
            error_urls.append(image_url)
            error_count += 1

    return np.array(image_embeddings), error_count

def apply_pca(image_embeddings, n_components=3):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(image_embeddings)

def apply_umap(image_embeddings, n_components=3):
    reducer = umap.UMAP(n_components=n_components)
    return reducer.fit_transform(image_embeddings)

def apply_tsne(image_embeddings, n_components=3):
    tsne = TSNE(n_components=n_components)
    return tsne.fit_transform(image_embeddings)

def apply_dbscan(embeddings, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(embeddings)
    return labels

def cluster_embeddings(embeddings, method='KMeans', n_clusters=5):
    if method == 'KMeans':
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(embeddings)
    elif method == 'DBSCAN':
        labels = apply_dbscan(embeddings)
    return labels

def visualize_embeddings(image_embeddings_transformed, labels, method, dataset, n_clusters):
    fig = px.scatter_3d(
        x=image_embeddings_transformed[:, 0],
        y=image_embeddings_transformed[:, 1],
        z=image_embeddings_transformed[:, 2],
        title=f"3D Scatter Plot of {method} Reduced Data",
        labels={'x': 'Component 1', 'y': 'Component 2', 'z': 'Component 3'},
        color=labels.astype(str),
        symbol=labels.astype(str),
        opacity=0.7
    )

    fig.update_traces(
        showlegend=True,
        selector=dict(type='scatter3d'),
        name='Cluster'
    )

    fig.update_layout(
        legend=dict(
            title='Cluster Labels',
            itemsizing='constant'
        )
    )

    st.plotly_chart(fig)

def compute_continuity(X, X_embedded, n_neighbors=5):
    og_dist = pairwise_distances(X)
    emb_dist = pairwise_distances(X_embedded)
    n_samples = X.shape[0]
    rank_og = np.argsort(np.argsort(og_dist, axis=1), axis=1)
    rank_emb = np.argsort(np.argsort(emb_dist, axis=1), axis=1)
    continuity_score = 0.0
    for i in range(n_samples):
        original_neighbors = np.argsort(og_dist[i])[:n_neighbors]
        embedded_neighbors = np.argsort(emb_dist[i])[:n_neighbors]
        diff = np.setdiff1d(original_neighbors, embedded_neighbors)
        for j in diff:
            continuity_score += (rank_emb[i, j] - n_neighbors)
    continuity_score = 1 - (2.0 / (n_samples * n_neighbors * (2 * n_samples - 3 * n_neighbors - 1))) * continuity_score
    return continuity_score

def compute_gdp(X, X_embedded, n_neighbors=5):
    isomap = Isomap(n_neighbors=n_neighbors, n_components=2)
    isomap.fit(X)
    geodesic_distances = isomap.dist_matrix_
    embedded_distances = pairwise_distances(X_embedded)
    stress = np.sum((geodesic_distances - embedded_distances)**2)
    original_sum = np.sum(geodesic_distances**2)
    return np.sqrt(stress / original_sum)

st.title("CLIP Image Embeddings")
st.sidebar.image("https://example.com/dsg_iitr_logo.jpg", use_column_width=True)
url = "https://huggingface.co/datasets/recastai/coyo-75k-augmented-captions"
st.sidebar.markdown(f"The dataset is a Subset of [COYO 400M Image-Text pairs]({url})")

if test_dataset is None:
    st.error("Failed to load the dataset. Please check your internet connection and try again.")
else:
    dim_reduction = st.selectbox("Choose Dimension Reduction Technique", ['PCA', 'UMAP', 'T-SNE'])
    clustering_algo = st.selectbox("Choose the clustering method", ['DBSCAN', 'K-MEANS'])
    n_cluster = st.slider("Number of clusters", 2, 10, 2) if clustering_algo == "K-MEANS" else None

    max_images = st.slider("Maximum number of images to process", 100, 1000, 500)
    
    image_embeddings, error_count = get_image_embeddings(test_dataset, max_images)

    if len(image_embeddings) > 0:
        if dim_reduction == "PCA":
            reduced_embeddings = apply_pca(image_embeddings)
        elif dim_reduction == "UMAP":
            reduced_embeddings = apply_umap(image_embeddings)
        elif dim_reduction == "T-SNE":
            reduced_embeddings = apply_tsne(image_embeddings)

        labels = cluster_embeddings(reduced_embeddings, method=clustering_algo, n_clusters=n_cluster)
        visualize_embeddings(reduced_embeddings, labels, clustering_algo, None, n_cluster)

        st.header("Comparison of Dimensionality Reduction Techniques")

        # Compute metrics for each technique
        trust_pca = trustworthiness(image_embeddings, apply_pca(image_embeddings))
        cont_pca = compute_continuity(image_embeddings, apply_pca(image_embeddings))
        gdp_pca = compute_gdp(image_embeddings, apply_pca(image_embeddings))

        trust_umap = trustworthiness(image_embeddings, apply_umap(image_embeddings))
        cont_umap = compute_continuity(image_embeddings, apply_umap(image_embeddings))
        gdp_umap = compute_gdp(image_embeddings, apply_umap(image_embeddings))

        trust_tsne = trustworthiness(image_embeddings, apply_tsne(image_embeddings))
        cont_tsne = compute_continuity(image_embeddings, apply_tsne(image_embeddings))
        gdp_tsne = compute_gdp(image_embeddings, apply_tsne(image_embeddings))

        # Display results
        st.subheader("Metrics")
        st.write("Trustworthiness (higher is better):")
        st.write(f"PCA: {trust_pca:.4f}")
        st.write(f"UMAP: {trust_umap:.4f}")
        st.write(f"t-SNE: {trust_tsne:.4f}")

        st.write("\nContinuity (higher is better):")
        st.write(f"PCA: {cont_pca:.4f}")
        st.write(f"UMAP: {cont_umap:.4f}")
        st.write(f"t-SNE: {cont_tsne:.4f}")

        st.write("\nGeodesic Distance Preservation (lower is better):")
        st.write(f"PCA: {gdp_pca:.4f}")
        st.write(f"UMAP: {gdp_umap:.4f}")
        st.write(f"t-SNE: {gdp_tsne:.4f}")

        # Determine the best technique
        best_trust = max(trust_pca, trust_umap, trust_tsne)
        best_cont = max(cont_pca, cont_umap, cont_tsne)
        best_gdp = min(gdp_pca, gdp_umap, gdp_tsne)

        st.subheader("Best Performing Technique")
        st.write(f"Best Trustworthiness: {best_trust:.4f}")
        st.write(f"Best Continuity: {best_cont:.4f}")
        st.write(f"Best Geodesic Distance Preservation: {best_gdp:.4f}")
    else:
        st.error("No valid image embeddings were generated. Please check your dataset.")

    st.write(f"Number of errors encountered: {error_count}")