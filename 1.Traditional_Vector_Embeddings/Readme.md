# Traditional Vector Embeddings

## Overview
Traditional vector embeddings are fundamental techniques used to represent textual and visual data in a dense vector space. These embeddings convert words, sentences, or images into high-dimensional vectors that capture their semantic meaning. This allows for various downstream tasks, including classification, clustering, and retrieval, by leveraging the relationships between these vectors.

### Examples of Traditional Vector Embeddings
- **Textual Embeddings:** Words or sentences are transformed into vectors. For instance, in the phrase "The cat sat on the mat," each word can be embedded into a vector space where similar words like "cat" and "mat" are positioned closer together.
- **Image Embeddings:** Images are processed through models like CLIP to produce vector representations that capture their visual features. This allows comparing images based on their content rather than raw pixel values.

## Challenges with Traditional Vector Embeddings
While traditional vector embeddings are powerful, they exhibit certain limitations, particularly when applied to Retrieval-Augmented Generation (RAG) or general Large Language Models (LLMs):

<img src= "https://github.com/AGAMPANDEYY/kge-clip-fork1/blob/main/media/vectoremb_db_llms.png" width= "600">

1. **Contextual Limitations:** Embeddings may lack contextual understanding, leading to inaccurate representations in complex scenarios.
2. **Dimensionality Issues:** High-dimensional vectors can be challenging to work with, leading to computational inefficiencies.
3. **Limited Generalization:** Embeddings trained on specific datasets may not generalize well to unseen data, reducing their applicability across diverse tasks.

## Models Used
In this repository, the following models were employed to generate text and image embeddings:
- **BERT:** A transformer-based model that creates contextual embeddings for sentences and words.
- **CLIP (Text and Image):** A multi-modal model that generates embeddings for both text and images, enabling cross-modal comparisons.
- **InferSent:** A model focused on sentence embeddings, designed to capture sentence-level semantics.
- **Universal Sentence Encoder (USE):** A model that provides general-purpose sentence embeddings for various tasks.

## Dimensionality Reduction Techniques

<img src="https://github.com/AGAMPANDEYY/kge-clip-fork1/blob/main/1.Traditional_Vector_Embeddings/3.Embedding_Scores/assets/UmapPCATsne.png" width="600">

Dimensionality reduction is employed to simplify high-dimensional embeddings, making them easier to visualize and interpret. The techniques used include:
- **PCA (Principal Component Analysis):** A linear technique that reduces dimensionality by projecting data onto the axes of maximum variance.
- **UMAP (Uniform Manifold Approximation and Projection):** A non-linear method that preserves the global structure of data, often leading to better clustering.
- **T-SNE (t-Distributed Stochastic Neighbor Embedding):** A non-linear technique designed for visualizing high-dimensional data in lower dimensions, focusing on preserving local structure.

### Comparison Techniques
To assess the quality of dimensionality reduction, the following metrics were used:
- **Linear Translation:** Evaluates how well the reduction technique preserves the linear relationships between vectors.
- **Cosine Similarity:** Measures the cosine of the angle between two vectors, indicating how similar they are in the reduced space.
- **Absolute Error:** Quantifies the difference between original and reduced vectors, providing insight into the accuracy of the reduction.

## Best Methodology and Data Storage
After extensive comparison, **UMAP** emerged as the best method for dimensionality reduction in this context due to its ability to balance global and local structure preservation. The embeddings and their reduced representations have been stored in CSV format within this repository, organized by the model and reduction technique used.

The results can be accessed in the respective subdirectories, with links provided in the summary table below:

| Sentence Encoder | PCA | UMAP | T-SNE |
|------------------|-----|------|-------|
| Infersent | [PCA](https://www.kaggle.com/datasets/anantjain1223/infersent-coyo-1k?select=infersent_pca.csv) | [UMAP](https://www.kaggle.com/datasets/anantjain1223/infersent-coyo-1k?select=infersent_umap.csv) | [T-SNE](https://www.kaggle.com/datasets/anantjain1223/infersent-coyo-1k?select=infersent_tsne.csv) |
| Clip-text | [PCA](https://www.kaggle.com/datasets/anantjain1223/clip-text-coyo-1k?select=clip_text_pca.csv) | [UMAP](https://www.kaggle.com/datasets/anantjain1223/clip-text-coyo-1k?select=clip_text_umap+csv.csv) | [T-SNE](https://www.kaggle.com/datasets/anantjain1223/clip-text-coyo-1k?select=clip_text_tsne.csv) |
| BERT | [PCA](https://www.kaggle.com/datasets/anantjain1223/sentence-transformer-coyo-1k?select=st_pca.csv) | [UMAP](https://www.kaggle.com/datasets/anantjain1223/sentence-transformer-coyo-1k?select=st_umap.csv) | [T-SNE](https://www.kaggle.com/datasets/anantjain1223/sentence-transformer-coyo-1k?select=st_tsne.csv) |
| Universal Sentence Encoder | [PCA](https://www.kaggle.com/datasets/anantjain1223/use-coyo-1k?select=use_pca.csv) | [UMAP](https://www.kaggle.com/datasets/anantjain1223/use-coyo-1k?select=use_umap.csv) | [T-SNE](https://www.kaggle.com/datasets/anantjain1223/use-coyo-1k?select=use_tsne.csv) |

## Conclusion
This repository provides a comprehensive analysis of traditional vector embeddings and their dimensionality reduction, offering insights into their applicability in various scenarios. The methodologies, models, and techniques documented here aim to assist researchers and practitioners in effectively utilizing and understanding embedding techniques.

For further details and results, please explore the respective folders and files in this repository.
