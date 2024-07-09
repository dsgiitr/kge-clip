import os
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from io import BytesIO
import requests
from sklearn.decomposition import PCA
import umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from transformers import CLIPProcessor, CLIPModel
from PIL import UnidentifiedImageError
import plotly.express as px
from datasets import load_dataset

# Set environment variables to address TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Streamlit app
st.title("CLIP Image Embeddings")
st.sidebar.text("The dataset is a 75k subset of COYO Image-Text pairs")

# Choose dimensionality reduction technique
dim_reduction = st.selectbox("Choose Dimension Reduction Technique", ['PCA', 'UMAP', 'T-SNE'])
clustering_algo = st.selectbox("Choose the clustering method", ['DBSCAN', 'K-MEANS'])
n_cluster = st.slider("Number of clusters", 2, 10, 2) if clustering_algo == "K-MEANS" else None

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load dataset
test_dataset = load_dataset("recastai/coyo-75k-augmented-captions")['test']

def plot_images_grid(images, text, grid_size=(5, 5), size=(5, 5)):
    nrows, ncols = grid_size
    fig, axes = plt.subplots(nrows, ncols, figsize=size)
    axes = axes.flatten()

    for idx, (img, txt) in enumerate(zip(images, text)):
        if isinstance(img, Image.Image):
            img = img.convert("RGB")
        axes[idx].imshow(img)
        axes[idx].set_title(txt, fontsize=10)
        axes[idx].axis('off')

    for ax in axes[len(images):]:
        ax.axis('off')

    plt.tight_layout()
    st.pyplot(fig)

dataframe = pd.DataFrame()

def get_image_embeddings(data):
    error_count = 0
    error_urls = []
    image_embeddings = []
    count = 0
    data_text = data['llm_caption']
    data_url = data['url']
    dataframe = pd.DataFrame({'url': data_url, 'text': data_text})

    for _, row in dataframe.iterrows():
        image_url = row['url']
        text = row['text'][0]
        try:
            count += 1
            if count == 20:
                break

            response = requests.get(image_url)
            response.raise_for_status()  # Raise an exception for non-2xx status codes
            image = Image.open(BytesIO(response.content))
            plot_images_grid([image], [text], grid_size=(1, 1))

            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                image_embedding = model.get_image_features(**inputs)
            image_embeddings.append(image_embedding.squeeze().cpu().numpy())

        except (requests.exceptions.RequestException, UnidentifiedImageError, ValueError) as e:
            error_urls.append(image_url)
            error_count += 1

    return image_embeddings, dataframe

def apply_pca(image_embeddings, n_components=3):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(image_embeddings)

def apply_umap(image_embeddings, n_components=3):
    umap_reducer = umap.UMAP(n_components=n_components)
    return umap_reducer.fit_transform(image_embeddings)

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
    fig.update_traces(showlegend=True, selector=dict(type='scatter3d'), name='Cluster')
    fig.update_layout(
        legend=dict(
            title='Cluster Labels',
            itemsizing='constant'
        )
    )
    st.plotly_chart(fig)

image_embeddings, dataframe = get_image_embeddings(test_dataset)

if dim_reduction == "PCA":
    reduced_embeddings = apply_pca(image_embeddings)
elif dim_reduction == "UMAP":
    reduced_embeddings = apply_umap(image_embeddings)
elif dim_reduction == "T-SNE":
    reduced_embeddings = apply_tsne(image_embeddings)

labels = cluster_embeddings(reduced_embeddings, method=clustering_algo, n_clusters=n_cluster)
visualize_embeddings(reduced_embeddings, labels, clustering_algo, dataframe, n_cluster)
