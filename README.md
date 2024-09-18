# Upgrading from Vectors to Graphs: Knowledge Graph Embeddings and Graph-RAG

![Knowledge Graph](https://github.com/AGAMPANDEYY/kge-clip-fork1/blob/main/media/KG_1.png)


## You can access the full project documentation on [Gitbook Link](https://agam-pandey.gitbook.io/knowledge-graph-embedding-or-dsg-iitr/)!

<div align="center">
  <img src="https://github.com/user-attachments/assets/02b2dab2-f1ba-4abe-a13a-ccb0b3791bb0" width="700" height="500"/>
</div>



### Project Highlights
- Explore **traditional vector embeddings** and their limitations.
- Discover how **CLIP** is used for image-text embeddings.
- Learn how **KGE** solves traditional embedding problems with a relational, graph-based approach.
- Experiment with **Graph-RAG** to improve reasoning and retrieval in **LLMs**.

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

> 📂 **/src**:  
> Core code for training Knowledge Graph Embeddings (KGE) using PyKeen, including scripts, configs, and data utilities.
>
> 📂 **/assets**:  
> Contains embedding results, visualizations, and key outputs from the models.
>
> 📑 **/notebooks**:  
> Jupyter notebooks for visualizing and comparing traditional and Knowledge Graph Embeddings (KGE).


## Setup Guide and Results

### 1. Access the dataset
- The dataset of 1k reduced COYO700M dataset can be found [Here](https://www.kaggle.com/datasets/anantjain1223/coyo-1k-reduced)
### 2. Creating Traditional Vector Embeddings 

4 Methods were used to create text embeddings and 1 CLIP notebook can be accessed for Image embeddings.

1. [CLIP Embeddings](https://github.com/dsgiitr/kge-clip/blob/main/1.Traditional_Vector_Embeddings/1.Text/embedding_models/clip_text.ipynb)
2. [InferSent Embeddings](https://github.com/dsgiitr/kge-clip/blob/main/1.Traditional_Vector_Embeddings/1.Text/embedding_models/infersent.ipynb)
3. [Universal Sentence Encoder](https://github.com/dsgiitr/kge-clip/blob/main/1.Traditional_Vector_Embeddings/1.Text/embedding_models/universal-sentence-encoder.ipynb)
4. [Bert](https://github.com/dsgiitr/kge-clip/blob/main/1.Traditional_Vector_Embeddings/1.Text/embedding_models/bert.ipynb)
5. [CLIP for Image Embeddings](https://github.com/dsgiitr/kge-clip/blob/main/1.Traditional_Vector_Embeddings/2.Images/embeddings_model/clip_vector_embeddings.ipynb)

| Step | Description |
|------|-------------|
| 1    | Open the eg.`CLIP_Embeddings.ipynb` notebook. |
| 2    | Run all the cells to load the CLIP model and generate embeddings. |
| 3    | Follow the instructions in the notebook to input your data and obtain embeddings. |


#### Requirements

- Python 3.x
- Required libraries (list them here)

#### How to Install

1. Clone the repository.
      ```bash
   git clone https://github.com/dsgiitr/kge-clip.git
   cd 1.Traditional_Vector_Embeddings
2. Install the required libraries using `pip install -r requirements.txt`.
3. Open the Jupyter notebooks and follow the instructions.
   
> [!TIP]
> [Refer to the Readme for more details on Traditional Vector Embeddings](https://github.com/dsgiitr/kge-clip/blob/main/1.Traditional_Vector_Embeddings/Readme.md)
### 3. Embeddings Visualization in 3D

<div align="center">
  <img src="https://github.com/user-attachments/assets/d05f9f65-e2d2-42fb-9499-4357123094a8" width="800" height="300"/>
</div>

To visualize text and image embeddings, use the following notebooks:

1. [Text Embeddings Visualizer](https://github.com/dsgiitr/kge-clip/blob/main/1.Traditional_Vector_Embeddings/1.Text/embeddings_visualisation/plotting_tensorboard.ipynb)
2. [Image Embeddings Visualizer](https://github.com/dsgiitr/kge-clip/blob/main/1.Traditional_Vector_Embeddings/2.Images/embeddings_visualisation/CLIP_Tensorboard_.ipynb)

Each embedding and cluster will be saved in `metadata.tsv`.

To launch TensorBoard, use:

```bash
%tensorboard --logdir /path/to/logs/embedding
```

### 4. Generating Knowledge Graphs

### Running the Neo4J Instance and Plotting Knowledge Graphs

Knowledge graphs were generated using the following steps:

1. **Triplet Extraction**  
   Run the `Rebel_extraction.ipynb` notebook to extract triplets using the [BabelScape REBEL-large](https://huggingface.co/Babelscape/rebel-large) model. You can find the notebook [here](https://github.com/dsgiitr/kge-clip/blob/main/2.Knowledge_Graphs/1.Text/codes/101-125/rebel.ipynb).

2. **Knowledge Graph Generation and Visualization**  
   Use the `KG.ipynb` notebook to generate knowledge graphs and visualize them using Neo4J, NetworkX, and Plotly. Access the notebook [here](https://github.com/dsgiitr/kge-clip/blob/main/2.Knowledge_Graphs/1.Text/codes/101-125/kg.ipynb).

<div align="center">
<img src= "https://github.com/user-attachments/assets/dc98c06f-702e-455f-925e-2b490dfa236b" width= 600 height=250/>
</div>


#### Running Neo4J Database Instance

To run a local Neo4J instance and visualize the knowledge graph:

1. **Install Neo4J**  
   Download and install Neo4J from the [official site](https://neo4j.com/download/).

2. **Start Neo4J**  
   After installation, start the Neo4J server:
   ```bash
   neo4j console

>[!TIP]
>[Refer to the Readme for more detail on Knowledge Graphs](https://github.com/dsgiitr/kge-clip/tree/main/2.Knowledge_Graphs)
### 5. Training Setup for PyKeen Knowledge Graph Embeddings
- Model: **TransE**
- Loss: **Softplus**
- Epochs: **100**
- Dataset: Triplets extracted from images and text using **REBEL** and **COYO Subset dataset**.
>[!TIP]
>[Refer to the Readme for more detail on KG_Embeddings](https://github.com/dsgiitr/kge-clip/tree/main/3.KG_Embeddings)
### 6. Storing Embeddings in FAISS index

### 7. Running the visualiser web-app


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
```

### Contributions

We welcome contributions to improve this project! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes with clear and descriptive messages.
4. Push the changes to your fork and submit a pull request.

>[!IMPORTANT]
>Please ensure your contributions align with the project's coding standards and include relevant documentation or tests. For major changes, consider opening an issue to discuss your approach first.

