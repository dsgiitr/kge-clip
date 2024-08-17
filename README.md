# Knowledge Graph Embeddings

Welcome to the **Knowledge Graph Embeddings KGE!**. 

This project is an exploration into how we can bridge the gap between text and image embeddings within a shared 3D space. Through this journey, we aim to discover the hidden relationships and contextual understandings that traditional methods might miss.

## Are Traditional Vector Embeddings even a problem!?

When we create vector embeddings for text and images using Large Language Models (LLMs) or generative models, the results often lack contextual depth. Imagine having a map but no sense of direction—that’s what these embeddings feel like! 

- Our goal is to bring clarity and meaning by leveraging the power of knowledge graphs.

## Here's the workflow we followed:

Here’s how we tackled the problem step-by-step:

1. **Dataset Preparation:**  
   We started with the COYO 700M text-image pair dataset and created a focused subset of 1,000 `{text: image}` pairs for our study.

2. **Embedding Generation:**  
   - **Text Embeddings:** Generated using `Universal Sentence Encoder (USE)`, `InferSent`, and `CLIP`.  
   - **Image Embeddings:** Generated using `CLIP`.
   - **Dimensionality Reduction:** Applied `PCA`, `T-SNE`, and `UMAP` to reduce the embeddings into a 3D space.


   ```python
   # Example: Generating text embeddings with USE
   import tensorflow as tf
   embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
   text_embedding = embed(["This is an example sentence."])

3. **Visualization & Analysis:** 
   We visualized these embeddings using TensorBoard and noticed a significant lack of context-aware embeddings, leading to poor correlation between text and image representations.

4. **Knowledge Graph Construction:**
   We constructed knowledge graphs using triplet extraction with REBEL and visualized them using NetworkX, Plotly, Graphviz, and Neo4j.
   ``` python
   # Example: Visualizing a knowledge graph with NetworkX
   import networkx as nx
   G = nx.Graph()
   G.add_edges_from([(1, 2), (2, 3)])
   nx.draw(G, with_labels=True)
   ```
5. **Embedding Knowledge Graphs:**

   Created embeddings for these knowledge graphs to assess their impact on contextual understanding.


## Insights and Future Scope
Through our analysis, we confirmed that without a knowledge graph, embeddings fail to capture the rich contextual relationships between text and images. With knowledge graph embeddings, we observed a marked improvement in the alignment and contextual relevance of embeddings in the 3D space.

KGE highlights the limitations of traditional embeddings from LLMs and generative models. By incorporating knowledge graph embeddings, we can achieve a much deeper and more accurate contextual understanding. Our next step? ### Exploring Graph Retrieval-Augmented Generation (RAG) to push the boundaries of generative models even further!
