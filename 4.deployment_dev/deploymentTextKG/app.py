from flask import Flask, request, render_template, jsonify
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import io
import torch
import base64
import math
import plotly
from matplotlib.figure import Figure
import plotly.graph_objs as go
from graphviz import Digraph

app = Flask(__name__)

# Load REBEL model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")

def token_gen():

    model_inputs = tokenizer(test_text,
                            max_length=512,
                            padding=True,
                            truncation=True,
                            return_tensors='pt')
    gen_kwargs = {
        "max_length": 216,
        "length_penalty": 0,
        "num_beams": 5,
        "num_return_sequences": 4
    }
    generated_tokens = model.generate(
        **model_inputs,
        **gen_kwargs,
    )
    decoded_preds = tokenizer.batch_decode(generated_tokens,
                                            skip_special_tokens=False)
    return decoded_preds

#extraction of triplets from model output so as to appear in head-relation-tail format
def extract_relations_from_model_output(text):
    relations = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    text_replaced = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")
    for token in text_replaced.split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                relations.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                relations.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        relations.append({
            'head': subject.strip(),
            'type': relation.strip(),
            'tail': object_.strip()
        })
    return relations

#Defining class for processing entire text
class KG():
    def __init__(self):
        self.relations = []

    def are_relations_equal(self, r1, r2):
        return all(r1[attr] == r2[attr] for attr in ["head", "type", "tail"])

    def exists_relation(self, r1):
        return any(self.are_relations_equal(r1, r2) for r2 in self.relations)

    def merge_relations(self, r1):
        r2 = [r for r in self.relations
              if self.are_relations_equal(r1, r)][0]
        spans_to_add = [span for span in r1["meta"]["spans"]
                        if span not in r2["meta"]["spans"]]
        r2["meta"]["spans"] += spans_to_add

    def add_relation(self, r):
        if not self.exists_relation(r):
            self.relations.append(r)
        else:
            self.merge_relations(r)

    def print(self):
        print("Relations:")
        for r in self.relations:
            print(f"  {r}")

    def save_csv(self,file_name):
        print(f"Saving to file {file_name}")
        reln_df = pd.DataFrame(self.relations)
        reln_df.to_csv(file_name,index=False)

#entire input text is tokenised here and processed at once
def from_small_text_to_KG(text,verbose=False):
    kg_instance=KG()

    dec_preds=token_gen()
    for sentence_pred in dec_preds:
        relations = extract_relations_from_model_output(sentence_pred)
        for r in relations:
            kg_instance.add_relation(r)

    return kg_instance

def from_large_text_to_KG(text, span_length=50, verbose=False):
    kg_instance=KG()
    inputs = tokenizer([text], return_tensors="pt")

    #compute span boundaries
    num_tokens = len(inputs["input_ids"][0])
    if verbose:
        print(f"Input has {num_tokens} tokens")
    num_spans = math.ceil(num_tokens / span_length)
    if verbose:
        print(f"Input has {num_spans} spans")
    overlap = math.ceil((num_spans * span_length - num_tokens) /
                        max(num_spans - 1, 1))
    spans_boundaries = []
    start = 0
    for i in range(num_spans):
        spans_boundaries.append([start + span_length * i,
                                 start + span_length * (i + 1)])
        start -= overlap
    if verbose:
        print(f"Span boundaries are {spans_boundaries}")

    #transform input with spans
    tensor_ids = [inputs["input_ids"][0][boundary[0]:boundary[1]]
                  for boundary in spans_boundaries]
    tensor_masks = [inputs["attention_mask"][0][boundary[0]:boundary[1]]
                    for boundary in spans_boundaries]
    inputs = {
        "input_ids": torch.stack(tensor_ids),
        "attention_mask": torch.stack(tensor_masks)
    }

    # generate relations
    num_return_sequences = 3
    gen_kwargs = {
        "max_length": 256,
        "length_penalty": 0,
        "num_beams": 3,
        "num_return_sequences": num_return_sequences
    }
    generated_tokens = model.generate(
        **inputs,
        **gen_kwargs,
    )

    # decode relations
    decoded_preds = tokenizer.batch_decode(generated_tokens,
                                           skip_special_tokens=False)

    i = 0
    for sentence_pred in decoded_preds:
        current_span_index = i // num_return_sequences
        relations = extract_relations_from_model_output(sentence_pred)
        for relation in relations:
            relation["meta"] = {
                "spans": [spans_boundaries[current_span_index]]
            }
            kg_instance.add_relation(relation)
        i += 1

    return kg_instance

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         text = request.form['text']
#         visualization_method = request.form['visualization']
#         kg = from_small_text_to_KG(text) if len(text) <= 100 else from_large_text_to_KG(text)
#         relation_df = pd.DataFrame(kg.relations)
        
#         if visualization_method == 'networkx':
#             graph_image = generate_networkx_graph(relation_df)
#         elif visualization_method == 'plotly':
#             graph_image = generate_plotly_graph(relation_df)
#         elif visualization_method == 'graphviz':
#             graph_image = generate_graphviz_graph(relation_df)
        
#         return render_template('result.html', relations=relation_df.to_dict('records'), graph_image=graph_image, method=visualization_method)
#     return render_template('index.html')

# # def generate_networkx_graph(relation_df):
# #     G = nx.DiGraph()
# #     for _, row in relation_df.iterrows():
# #         G.add_edge(row['head'], row['tail'], type=row['type'])
    
# #     plt.figure(figsize=(12, 8))
# #     pos = nx.spring_layout(G)
# #     nx.draw(G, pos, with_labels=True, node_color='#00008B', node_size=3000, font_size=10, font_color='white')
# #     edge_labels = nx.get_edge_attributes(G, 'type')
# #     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
# #     img = io.BytesIO()
# #     plt.savefig(img, format='png', facecolor='#808080', edgecolor='none')
# #     img.seek(0)
# #     graph_url = base64.b64encode(img.getvalue()).decode()
# #     plt.close()
# #     return f'data:image/png;base64,{graph_url}'
# def generate_networkx_graph(relation_df):
#     # Create a directed graph
#     G = nx.from_pandas_edgelist(relation_df, source='head', target='tail', edge_attr='type', create_using=nx.DiGraph())

#     # Create a new figure
#     fig = Figure(figsize=(8, 6))
#     ax = fig.add_subplot(111)

#     # Generate layout for nodes
#     pos = nx.spring_layout(G)

#     # Draw nodes and edges
#     nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='black', node_size=2000, ax=ax, font_size=10, font_weight='bold')

#     # Draw edge labels
#     edge_labels = nx.get_edge_attributes(G, 'type')
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', ax=ax)

#     # Remove axes
#     ax.axis('off')

#     # Save the figure to a bytes buffer
#     buf = io.BytesIO()
#     fig.savefig(buf, format='png')
#     buf.seek(0)

#     # Encode the image in base64
#     graph_url = base64.b64encode(buf.getvalue()).decode('utf-8')

#     # Return a data URL
#     return f'data:image/png;base64,{graph_url}'

# def generate_plotly_graph(relation_df):
#     G = nx.from_pandas_edgelist(relation_df, source='head', target='tail', edge_attr=True, create_using=nx.DiGraph())
#     pos = nx.spring_layout(G)
    
#     edge_x = []
#     edge_y = []
#     for edge in G.edges():
#         x0, y0 = pos[edge[0]]
#         x1, y1 = pos[edge[1]]
#         edge_x.extend([x0, x1, None])
#         edge_y.extend([y0, y1, None])

#     edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

#     node_x = [pos[node][0] for node in G.nodes()]
#     node_y = [pos[node][1] for node in G.nodes()]

#     node_trace = go.Scatter(
#         x=node_x, y=node_y, mode='markers+text', hoverinfo='text',
#         marker=dict(showscale=True, colorscale='YlGnBu', size=10, color=[], colorbar=dict(thickness=15, title='Node Connections')),
#         text=[node for node in G.nodes()], textposition="top center"
#     )

#     fig = go.Figure(data=[edge_trace, node_trace],
#                     layout=go.Layout(showlegend=False, hovermode='closest',
#                                      margin=dict(b=20,l=5,r=5,t=40),
#                                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#                                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

#     graph_json = plotly.io.to_json(fig)
#     return graph_json

# def generate_graphviz_graph(relation_df):
#     dot = Digraph(comment='Knowledge Graph')
#     dot.attr(rankdir='LR', size='8,5', bgcolor='white')
#     dot.attr('node', style='filled', color='#1E90FF', fontcolor='white')
#     dot.attr('edge', color='#666666')

#     for _, row in relation_df.iterrows():
#         dot.node(row['head'])
#         dot.node(row['tail'])
#         dot.edge(row['head'], row['tail'], label=row['type'])

#     graph_svg = dot.pipe(format='svg').decode('utf-8')
#     return graph_svg

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        visualization_method = request.form['visualization']
        kg = from_small_text_to_KG(text) if len(text) <= 100 else from_large_text_to_KG(text)
        relation_df = pd.DataFrame(kg.relations)
        
        if visualization_method == 'networkx':
            graph_image = generate_networkx_graph(relation_df)
        elif visualization_method == 'plotly':
            graph_image = generate_plotly_graph(relation_df)
        elif visualization_method == 'graphviz':
            graph_image = generate_graphviz_graph(relation_df)
        
        return render_template('result.html', relations=relation_df.to_dict('records'), graph_image=graph_image, method=visualization_method)
    return render_template('index.html')

def generate_networkx_graph(relation_df):
    G = nx.DiGraph()
    for _, row in relation_df.iterrows():
        G.add_edge(row['head'], row['tail'], type=row['type'])
    
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='#1E90FF', node_size=3000, font_size=10, font_weight='bold', font_color='white')
    edge_labels = nx.get_edge_attributes(G, 'type')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    img = io.BytesIO()
    plt.savefig(img, format='png', facecolor='white', edgecolor='none')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return f'data:image/png;base64,{graph_url}'

def generate_plotly_graph(relation_df):
    G = nx.from_pandas_edgelist(relation_df, source='head', target='tail', edge_attr=True, create_using=nx.DiGraph())
    pos = nx.spring_layout(G)
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text', hoverinfo='text',
        marker=dict(showscale=True, colorscale='YlGnBu', size=10, color=[], colorbar=dict(thickness=15, title='Node Connections')),
        text=[node for node in G.nodes()], textposition="top center"
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(showlegend=False, hovermode='closest',
                                     margin=dict(b=20,l=5,r=5,t=40),
                                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    graph_json = plotly.io.to_json(fig)
    return graph_json

def generate_graphviz_graph(relation_df):
    dot = Digraph(comment='Knowledge Graph')
    dot.attr(rankdir='LR', size='8,5', bgcolor='white')
    dot.attr('node', style='filled', color='#1E90FF', fontcolor='white')
    dot.attr('edge', color='#666666')

    for _, row in relation_df.iterrows():
        dot.node(row['head'])
        dot.node(row['tail'])
        dot.edge(row['head'], row['tail'], label=row['type'])

    graph_svg = dot.pipe(format='svg').decode('utf-8')
    return graph_svg

if __name__ == '__main__':
    app.run(debug=True)