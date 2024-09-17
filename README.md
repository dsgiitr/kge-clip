# Upgrading from Vectors to Graphs: Knowledge Graph Embeddings and Graph-RAG

![Knowledge Graph](https://github.com/AGAMPANDEYY/kge-clip-fork1/blob/main/media/KG_1.png)


## The detailed documentation of the KGE Project can be found at the [Gitbook Link](https://agam-pandey.gitbook.io/knowledge-graph-embedding-or-dsg-iitr/)

<div align="center">
  <img src="https://github.com/user-attachments/assets/03bc07b6-be37-4b61-9316-a6f8613e66d1" width="700" height="500"/>
</div>

In machine learning, the shift from traditional vectors to graph-based structures is transformative. Knowledge Graph Embeddings (KGE) offer a powerful way to represent relationships between entities, enhancing tasks like entity linking, relationship extraction, and semantic search.

This project explores KGE and introduces Graph-RAG, a method that enhances reasoning and retrieval in graph databases.

### Project Highlights
- Explore **traditional vector embeddings** and their limitations.
- Discover how **CLIP** is used for image-text embeddings.
- Learn how **KGE** solves traditional embedding problems with a relational, graph-based approach.
- Experiment with **Graph-RAG** to improve reasoning and retrieval in **LLMs**.
  
## Repository Overview

- 📁 **[Repository Link](https://github.com/dsgiitr/kge-clip.git)**
  This repository contains the code, models, and resources related to the exploration of Knowledge Graph Embeddings (KGE) and traditional vector embeddings for text and image data.

### Directory Structure
    .
    ├── 1_Traditional_Vector_Embeddings   # Traditional text and image embeddings using Word2Vec and CLIP
    ├── 2_Knowledge_Graphs                # Code and resources for generating Knowledge Graphs and extracting triplets
    ├── 3_KG_Embeddings                   # Knowledge Graph Embeddings (KGE) training using PyKeen and dimensionality reduction
    ├── 4_Deployment_dev                  # Scripts for deploying and testing embedding models
    ├── 6_FAISS_embeddings                # FAISS-based search for efficient embedding retrieval and comparisons
    └── README.md                         # Project documentation

> Follow the directories to get src, assets for image and text datasets

### Additional Directories

- 📂 **/src**:  
  The core code for training Knowledge Graph Embeddings (KGE) using the PyKeen framework. This folder includes model training scripts, configurations, and utilities for data processing.

- 📂 **/assets**:  
  A collection of embedding result files, visualizations, and other assets generated during the project. This directory contains key outputs from the embedding models, including reduced-dimensionality visualizations.

- 📑 **/notebooks**:  
  Jupyter notebooks for visualizing and comparing traditional vector embeddings (Word2Vec, CLIP) and Knowledge Graph Embeddings. These notebooks guide you through the results and offer insights into the differences between embedding approaches.


## Setup Guide and Results

1. [Refer to the Readme for more details on Traditional Vector Embeddings](https://github.com/dsgiitr/kge-clip/blob/main/1.Traditional_Vector_Embeddings/Readme.md)
2. [Refer to the Readme for more detail on Knowledge Graphs](https://github.com/dsgiitr/kge-clip/tree/main/2.Knowledge_Graphs)
3. [Refer to the Readme for more detail on KG_Embeddings](https://github.com/dsgiitr/kge-clip/tree/main/3.KG_Embeddings)
4. [Check out the development_dev for accesing the visualiser App]()
#### Training Setup
- Model: **TransE**
- Loss: **Softplus**
- Epochs: **100**
- Dataset: Triplets extracted from images and text using **REBEL** and **COYO Subset dataset**.

### Image and Text KG Embeddings
- **Image Triplets**: Extracted using REBEL from image URLs.
  - [Dataset Link](https://www.kaggle.com/datasets/agampy/triplets-kg)
- **Text Triplets**: Extracted using REBEL-large.
  - [Dataset Link](https://www.kaggle.com/datasets/agampy/text-triplets1k)

For detailed code and model configurations, refer to our notebooks:
- [PyKeen for Image KG Embeddings](https://github.com/dsgiitr/kge-clip/blob/main/3.KG_Embeddings/src/pykeen_KGE.ipynb)
- [PyKeen for Text KG Embeddings](https://github.com/dsgiitr/kge-clip/blob/main/3.KG_Embeddings/src/pykeen_KGE_text.ipynb)

## Results and Comparisons

### Comparison with Traditional Vector Embeddings
We compared traditional vector embeddings (Word2Vec, CLIP) with KGE across text and images. By reducing embeddings to 3D space using **PCA**, **UMAP**, and **t-SNE**, we explored how different embeddings capture context from the dataset. Results were visualized in **TensorBoard** to better understand vector representation across different models.

- 📂 **Results Folder**: Find the embedding result files [here](3.KG_Embeddings/assets/results/reduced_embeddings).

## Visualizer App and User Guide
We’ve developed a visualizer to compare text and image embeddings:
- **Text Embeddings**: Visualize and compare embeddings using Streamlit.
- **Image Embeddings**: Analyze image embeddings in the same space as text.
- **KG Generation App**: Generate and explore Knowledge Graphs interactively.

## Getting Started

### Prerequisites
- Python 3.x
- Libraries: `PyTorch`, `PyKeen`, `t-SNE`, `PCA`, `UMAP`, `Streamlit`

### Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/dsgiitr/kge-clip.git
cd kge-clip
pip install -r requirements.txt
