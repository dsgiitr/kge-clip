# Upgrading from Vectors to Graphs: Knowledge Graph Embeddings and Graph-RAG

![Knowledge Graph](https://github.com/AGAMPANDEYY/kge-clip-fork1/blob/main/media/KG_1.png)


## Introduction
In the evolving landscape of machine learning, moving from traditional vector-based representations to more intricate graph structures marks a significant shift. **Knowledge Graph Embeddings (KGE)** provide a powerful way to represent relationships between entities, surpassing the limitations of static vectors. These embeddings are critical for tasks like entity linking, relationship extraction, and semantic search where understanding the context is crucial.

This project delves into **Knowledge Graph Embeddings** and introduces **Graph-RAG**, a cutting-edge method that leverages these embeddings to enable more nuanced reasoning and retrieval across graph databases.

### Project Highlights
- Explore **traditional vector embeddings** and their limitations.
- Discover how **CLIP** is used for image-text embeddings.
- Learn how **KGE** solves traditional embedding problems with a relational, graph-based approach.
- Experiment with **Graph-RAG** to improve reasoning and retrieval in **LLMs**.
  
## Repository Overview

- 📁 **[Repository Link](https://github.com/dsgiitr/kge-clip.git)**
  This repository contains the code, models, and resources related to the exploration of Knowledge Graph Embeddings (KGE) and traditional vector embeddings for text and image data.

### Directory Structure

- 📊 **/1. Traditional Vector Embeddings**:  
  Contains implementations and results related to traditional text and image embeddings using models like Word2Vec and CLIP. Useful for understanding baseline performance in comparison to Knowledge Graph Embeddings.

- 🧠 **/2. Knowledge Graphs**:  
  Contains code and resources for generating and working with Knowledge Graphs. Includes extracted triplets and graph-based representations for both text and image data.

- 🔗 **/3. KG_Embeddings**:  
  Focuses on training and evaluating Knowledge Graph Embeddings (KGE) using the PyKeen library. This includes configurations for different models (like TransE), loss functions, and dimensionality reduction techniques like PCA, UMAP, and t-SNE.

- 🚀 **/4. Deployment_dev**:  
  Code related to deployment and testing of models in a development environment. Contains scripts and configurations to set up and deploy embedding models in a production or testing environment.

- 🔍 **/6. FAISS_embeddings**:  
  Includes FAISS-based nearest neighbor search for efficiently comparing and retrieving embeddings. FAISS is used here to index and search embeddings for large-scale similarity comparisons between text and image data.

### Additional Directories

- 📂 **/src**:  
  The core code for training Knowledge Graph Embeddings (KGE) using the PyKeen framework. This folder includes model training scripts, configurations, and utilities for data processing.

- 📂 **/assets**:  
  A collection of embedding result files, visualizations, and other assets generated during the project. This directory contains key outputs from the embedding models, including reduced-dimensionality visualizations.

- 📑 **/notebooks**:  
  Jupyter notebooks for visualizing and comparing traditional vector embeddings (Word2Vec, CLIP) and Knowledge Graph Embeddings. These notebooks guide you through the results and offer insights into the differences between embedding approaches.

  
## Traditional Vector Embeddings

### What are Vector Embeddings?
Transformer-based models like **BERT**, **GPT**, and **T5** leverage self-attention mechanisms to process text and generate context-specific embeddings. In this project, we explore these embeddings using **Word2Vec** for text and **CLIP** for image embeddings.

### Image Embeddings with CLIP
- **Process**: Images are represented as tensors, processed by the pre-trained CLIP model to generate compact 1D vectors.
- **Dimensionality Reduction**: Embeddings are reduced to 3D using **t-SNE** for visual analysis.
  
We analyze embeddings to understand how image representations compare to text in tasks like image search, captioning, and cross-modal analysis.

## The Traditional Embeddings Problem
Traditional embeddings often lack the ability to capture complex, multi-modal relationships between images and text. For example, embeddings of semantically similar data, such as fitness equipment and home decor, can be erroneously clustered with unrelated objects due to vector space limitations.

### The Solution: Knowledge Graph Embeddings (KGE)
By representing relationships in a **Knowledge Graph**, we move beyond isolated vectors. KGEs allow us to model deeper, more meaningful connections between entities and their contexts, which enhances tasks like **information retrieval** and **recommendations** in **LLMs**.

## Knowledge Graph Generation

### What is Graph Representation?
We use **Relationformer**, an encoder-decoder architecture trained on the **Visual Genome dataset**, to generate scene graphs from images. This approach captures object and relationship tokens for tasks like object detection and relation prediction.

### KG Embeddings with PyKeen
We utilized the **PyKeen** library to create and train **TransE** models for Knowledge Graph embeddings. Both image and text triplets were processed to generate embeddings.

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
