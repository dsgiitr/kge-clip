# Dimensionality Reduction Techniques Comparison

## Overview

This project provides a mathematical comparison of various dimensionality reduction techniques—such as PCA, UMAP, and t-SNE—on different datasets. The objective is to identify the most suitable technique for a given dataset based on its unique characteristics, such as noise levels, dimensionality, and underlying structures.

### Why Compare Techniques?

Different datasets have varying characteristics, and similarly, dimensionality reduction techniques come with distinct underlying assumptions, goals, and mathematical formulations. By mathematically comparing these techniques, we can understand how they behave under different conditions, ensuring that the selected method aligns with the nature of the data.

Additionally, there is an inherent trade-off between preserving local versus global structures, computational efficiency, and interpretability in these dimension reduction techniques. Mathematical comparisons reveal how these theoretical properties translate into practice, empowering informed decisions about which technique is most appropriate for specific data, objectives, and constraints.

## coyo-1k-TEXT Dataset

The coyo-1k-TEXT dataset is one of the datasets used for comparison in this project. It is specifically designed to evaluate how well different dimensionality reduction techniques perform in terms of local and global structure preservation, geometric accuracy, and computational efficiency.

### Scores for coyo-1k-TEXT Dataset

Below are the comparison scores of PCA, UMAP, and t-SNE techniques on the coyo-1k-TEXT dataset:

| Comparison                     | PCA          | UMAP      | t-SNE     |
| ------------------------------ | ------------ | --------- | --------- |
| Trustworthiness                | 0.999700     | 0.999543  | 0.999591  |
| Continuity                     | 0.999799     | 0.999736  | 0.999751  |
| Cosine Similarity              | 7.257074e-07 | 0.150968  | 0.002865  |
| Linear Translation             | 1.319569e-25 | 22.482631 | 30.238292 |
| Euclidean Correlation          | 0.999997     | 0.738646  | 0.981130  |
| Geodesic Distance Preservation | 0.572656     | 0.982638  | 0.948360  |

## Key Metrics Explained

- **Trustworthiness:** Measures local structure preservation. **Higher** scores indicate better preservation, with values ranging from 0 to 1.
- **Continuity:** Measures global structure preservation by embeddings. **Higher** scores indicate better preservation, with values ranging from 0 to 1.
- **Geodesic Distance Preservation (GDP):** Focuses on preserving the intrinsic geometry of the manifold. **Lower** values indicate better preservation.
- **Cosine Similarity:** Indicates the preservation of angular relationships. **Higher** similarity implies better preservation.
- **Linear Translation (MSE):** Measures the mean squared error (MSE) of the mapping. **Lower** values indicate better mapping.
- **Euclidean Correlation:** Measures the linear correlation between the original and reduced spaces. **Higher** correlation indicates better technique performance.
