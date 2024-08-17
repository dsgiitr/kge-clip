# Knowledge Graph Embeddings

Welcome to the **Knowledge Graph Embeddings KGE !**. 

This project is an exploration into how we can bridge the gap between text and image embeddings within a shared 3D space. Through this journey, we aim to discover the hidden relationships and contextual understandings that traditional methods might miss. Let’s dive in! 🚀

## Are Vector Embeddings even a problem!?

When we create vector embeddings for text and images using Large Language Models (LLMs) or generative models, the results often lack contextual depth. Imagine having a map but no sense of direction—that’s what these embeddings feel like! 

- Our goal is to bring clarity and meaning by leveraging the power of knowledge graphs.

## Methodology 

Here’s how we tackled the problem step-by-step:

1. **📂 Dataset Preparation:**  
   We started with the COYO 700M text-image pair dataset and created a focused subset of 1,000 `{text: image}` pairs for our study.

2. **🔢 Embedding Generation:**  
   - **Text Embeddings:** Generated using `Universal Sentence Encoder (USE)`, `InferSent`, and `CLIP`.  
   - **Image Embeddings:** Generated using `CLIP`.
   - **Dimensionality Reduction:** Applied `PCA`, `T-SNE`, and `UMAP` to reduce the embeddings into a 3D space.

   ```python
   # Example: Generating text embeddings with USE
   import tensorflow as tf
   embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
   text_embedding = embed(["This is an example sentence."])
