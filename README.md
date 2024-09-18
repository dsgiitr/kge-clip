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


----------------------------------------------------------------------------------------------------------------------------------------
  
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


----------------------------------------------------------------------------------------------------------------------------------------

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

----------------------------------------------------------------------------------------------------------------------------------------

### 4. Generating Knowledge Graphs

Knowledge graphs foe both `{text:image}` pairs were generated using the following steps:

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
Run the following code snippet to set up a Neo4J database remotely after setting up an account.

```python
from neo4j import GraphDatabase

# Connect to Neo4j
uri = "neo4j+s://647567ec.databases.neo4j.io"  # Replace with your Neo4j instance URI
username = "neo4j"
password = "mnx05CnETPwiMvSG7vQBZQwvJLz951fKhX-3zDfNVQg"  # Replace with your Neo4j password
driver = GraphDatabase.driver(uri, auth=(username,password))

def create_nodes_and_relationships(tx, head, type_, tail):
    query = (
        "MERGE (a:head {name: $head}) "
        "MERGE (b: tail {name: $tail}) "
        "MERGE (a)-[r : Relation {type: $type}]->(b)"
    )
    tx.run(query, head=head, type=type_, tail=tail)

#df_rebel_text=df_rebel['triplet'].tolist()
# Open a session and add data
with driver.session() as session:
    for row in triplets:
        session.write_transaction(create_nodes_and_relationships, row['head'], row['type'], row['tail'])

print("Knowledge graph created successfully!")

driver.close()
```
4. Run the following CyPhwer query on Neo4J Database instance:
```bash
MATCH (n)-[r]->(m)
RETURN n, r, m
```
>[!TIP]
>[Refer to the Readme for more detail on Knowledge Graphs](https://github.com/dsgiitr/kge-clip/tree/main/2.Knowledge_Graphs)


----------------------------------------------------------------------------------------------------------------------------------------

### 5. PyKeen Knowledge Graph Embedding Training

The PyKeen model is trained on Text and Image KG triplets extracted using `Babelscape REBEL-large`.

- Access the text KGE notebook: [`pykeen_KGE_text.ipynb`](https://github.com/dsgiitr/kge-clip/blob/main/3.KG_Embeddings/src/pykeen_KGE_text.ipynb)
- Access the image KGE notebook: [`pykeen_KGE_Image.ipynb`](https://github.com/dsgiitr/kge-clip/blob/main/3.KG_Embeddings/src/pykeen_KGE_Image.ipynb)

#### PyKeen Model Configuration

```python
from pykeen.pipeline import pipeline

result = pipeline(
    model='TransE',  # Choose a graph embedding technique
    loss="softplus",
    training=training_triples_factory,
    testing=testing_triples_factory,
    model_kwargs=dict(embedding_dim=3),  # Set embedding dimensions
    optimizer_kwargs=dict(lr=0.1),  # Set learning rate
    training_kwargs=dict(num_epochs=100, use_tqdm_batch=False),  # Set number of epochs
)
```


The trained KGE for both text and Image are further reduced to 3D space using PCA/UMAP & t-SNE.
Result embeddings and media can be found in the `assets` folder [here](https://github.com/dsgiitr/kge-clip/tree/main/3.KG_Embeddings/assets/results)


>[!TIP]
>[Refer to the Readme for more detail on KG_Embeddings](https://github.com/dsgiitr/kge-clip/tree/main/3.KG_Embeddings)


----------------------------------------------------------------------------------------------------------------------------------------

### 6. Storing Embeddings in FAISS index

FAISS database was used to store the `{text:image}` Vector and Knowledge Graph embeddings for using it further with RAG-LLMs

Access the FAISS index notebook [here](https://github.com/dsgiitr/kge-clip/blob/main/6.FAISS-embeddings/src/FAISS_Embeddings.ipynb)
Set the dimensions as per what the LLM model needs.

```python
import faiss

dimension=512
index=faiss.IndexFlatL2(dimension)

index.add(embeddings_img_array) #add the img embedding in faiss
index.add(embeddings_text_array) # add text embedding in faiss

faiss.write_index(index, 'faiss_traditional_vector_embedding.index')
```

----------------------------------------------------------------------------------------------------------------------------------------

### 7. Running the KG visualiser web-app

This repository contains a Flask-based web app that supports:

- **Text-Based Knowledge Graph Generation**
- **Image-Based Knowledge Graph Generation**
- **Text & Image Vector Embedding and Knowledge Graph Embedding with TensorBoard**

The app utilizes Python libraries, the REBEL model, and Graphviz for advanced graph visualization.

<div align="center">
  <img src= "https://github.com/user-attachments/assets/0ae1b3b9-e0ba-4166-93d6-5cde4f4d4ed5" width=600 height=300/>
</div>

Follow these steps to set up and run the web app.

**Prerequisites**

Ensure your environment meets the following requirements:

1. Python 3.7 or higher
2. `pip` (Python package installer)
3. [Graphviz](https://graphviz.org/) for advanced graph visualization

**Installation**

1. **Clone the Repository**

Fork the project and clone it to your local machine:

```bash
git clone https://github.com/dsgiitr/kge-clip.git
cd kge-clip/deployment_dev
```

Set Up and Run the Flask App. Activate a virtual environment to manage dependencies:

- **On Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```
- **On macOS/Linux:**
``` bash
python3 -m venv venv
source venv/bin/activate
```

Install Dependencies
Install the required Python packages:

``` bash
pip install flask transformers torch pandas networkx matplotlib plotly graphviz
```

Running the Flask App
Activate the Virtual Environment and start the Flask App.

- **On Windows:**

``` bash
venv\Scripts\activate
set FLASK_APP=app.py

``` 
- **On macOS/Linux:**

``` bash
source venv/bin/activate
export FLASK_APP=app.py
```

Run the Flask app with:

```bash
flask run
```

Open your web browser and navigate to `http://127.0.0.1:5000/` to start using the app.


----------------------------------------------------------------------------------------------------------------------------------------

## Results and Comparisons

>[!NOTE]
> Detailed result and descriptions are explained in the [DSG Gitbook](https://agam-pandey.gitbook.io/knowledge-graph-embedding-or-dsg-iitr/the-traditional-embeddings-problem)

The results were divied into 

1. Traditional Vector embeddings 3D Reduced visualisation using Tensorboard. [📂 **Results Folder**]( https://github.com/dsgiitr/kge-clip/blob/main/1.Traditional_Vector_Embeddings/1.Text/embeddings_visualisation/plotting_tensorboard.ipynb)
2. Similarity scores of reduced embeddings of different Text encoder.  [📂 **Results Folder**](https://github.com/dsgiitr/kge-clip/tree/main/1.Traditional_Vector_Embeddings/3.Embedding_Scores)
3. Comparing image and text vector embeddings disparity and contextual drawbacks.  [📂 **Results Folder**](https://agam-pandey.gitbook.io/knowledge-graph-embedding-or-dsg-iitr/the-traditional-embeddings-problem)
4. Scene Graph Generation of {text:image} pair using VLM & Relationformer.  [📂 **Results Folder**](https://github.com/dsgiitr/kge-clip/blob/main/2.Knowledge_Graphs/2.Images/code/2.%20Relationformer.ipynb) 
5. KG Visualisation with Neo4j, NetworkX, Plotly and Graphviz.  [📂 **Results Folder**](https://github.com/dsgiitr/kge-clip/tree/main/2.Knowledge_Graphs/1.Text/visualisation/KG_Visualisation151-200)
6. KG and traditional vector Embeddings .csv  [📂 **Results Folder**](https://github.com/dsgiitr/kge-clip/tree/main/3.KG_Embeddings/assets/results)
---------------------------------------------------------------------------------------------------------------------------------------

### Core Contributors
The list of core contributors to this repository are (mentioned alphabetically):

- [Aastha Khaitan](https://github.com/AK1405)
- [Advika Sinha](https://github.com/advikasinha)
- [Agam Pandey](https://github.com/AGAMPANDEYY)
- [Anant Jain](https://github.com/anant37289)
- [Simardeep Singh](https://github.com/heyysimarr)

---------------------------------------------------------------------------------------------------------------------------------------

### Contributions 🚀

We welcome contributions to improve this project! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes with clear and descriptive messages.
4. Push the changes to your fork and submit a pull request.

>[!IMPORTANT]
>Please ensure your contributions align with the project's coding standards and include relevant documentation or tests. For major changes, consider opening an issue to discuss your approach first.

