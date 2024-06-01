
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
from sklearn.cluster import KMeans,DBSCAN # Kmeans for Centroidal Cluserting while DBSCAN for density based
from transformers import CLIPProcessor,CLIPModel
from PIL import UnidentifiedImageError
import plotly.express as px
from tqdm import tqdm
import matplotlib.pyplot as plt
#import supervision as sv
import random

device="cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor=CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

from datasets import load_dataset

#dataset=load_dataset("isidentical/moondream2-coyo-5M-captions")['train']
n=100 #number of samples
#dataset=dataset[:n]


dataset = load_dataset("isidentical/moondream2-coyo-5M-captions")['train']
dataset=dataset[:n]

def plot_images_grid(images, grid_size=(5, 5), titles=None, size=(5, 5), cmap=None):
    from math import ceil
    
    nrows, ncols = grid_size
    fig, axes = plt.subplots(nrows, ncols, figsize=size)
    
    if nrows == 1 and ncols == 1:
        axes = [axes]  # Make axes iterable if it's a single Axes object
    else:
        axes = axes.flatten()
    
    for idx, img in enumerate(images):
        if isinstance(img, Image.Image):
            img = img.convert("RGB")
        if cmap:
            axes[idx].imshow(img, cmap=cmap)
        else:
            axes[idx].imshow(img)
        
        if titles:
            axes[idx].set_title(titles[idx], fontsize=10)
        
        axes[idx].axis('off')
    

    for ax in axes[len(images):]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def get_image_embeddings(image):
    error_count = 0
    error_urls = []
    image_embeddings = [] 
    countn=0
    for example in tqdm(data):

        image_url = example['url']
        try:

            coutn+=1

            if coutn == 50:
                break

            response = requests.get(image_url)
            response.raise_for_status()  # Raise an exception for non-2xx status codes

            # Check if the content type is an image
            image=Image.open(BytesIO(response.content))
            plot_images_grid([image], grid_size=(1, 1))


            inputs=processor(images=image,return_tensors="pt").to(device)
            with torch.no_grad():
                image_embedding=model.get_image_features(**inputs)
            image_embeddings.append(image_embedding.squeeze().cpu().numpy())

        except (requests.exceptions.RequestException, UnidentifiedImageError, ValueError) as e:
            error_urls.append(image_url)
            error_count += 1

    return image_embeddings, error_count
    
    

    
def apply_pca(image_embeddings,n_components=3):
    pca=PCA(n_components=n_components)
    return pca.fit_transform(image_embeddings)

def apply_umap(image_embeddings,n_components=3):
    umap=umap.UMAP(n_components=n_components)
    return umap.fit_transform(image_embeddings)

def apply_tsne(image_embeddings,n_components=3):
    tsne=TSNE(n_components=n_components)
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

def visualize_embeddings(image_embeddings_transformed,labels,method,dataset,n_clusters):

    plt.figure(figsize=(10, 10))
    fig = px.scatter_3d(
    x=image_embeddings_transformed[:, 0],
    y=image_embeddings_transformed[:, 1],
    z=image_embeddings_transformed[:, 2],
    title= f"3D Scatter Plot of {method} Reduced Data",
    labels={'x': 'Component 1', 'y': 'Component 2', 'z': 'Component 3'},
    hover_data=dataset['moondream2_caption']
    )
    plt.legend(handles=fig.legend_elements()[0], labels=set(labels))
    st.plotly_chart(plt)



st.title("CLIP Image Embeddings")
st.sidebar.text("The dataset is Subset of COYO 400M Image-Text pairs")
dim_reduction=st.selectbox("Choose Dimension Reduction Technique",['PCA','UMAP','T-SNE'])
clustering_algo=st.selectbox("Choose the clustering method", ['DBSCAN','K-MEANS'])
n_cluster=st.slider("Number of clusters", 2,10,2) if clustering_algo=="K-MEANS" else None

image_embeddings,error_count=get_image_embeddings(dataset)


if dim_reduction == "PCA":
        reduced_embeddings = apply_pca(image_embeddings)
elif dim_reduction == "UMAP":
        reduced_embeddings = apply_umap(image_embeddings)
elif dim_reduction == "T-SNE":
        reduced_embeddings = apply_tsne(image_embeddings)

labels = cluster_embeddings(reduced_embeddings, method=clustering_algo, n_clusters=n_cluster,dataset=dataset)
visualize_embeddings(reduced_embeddings, labels)
