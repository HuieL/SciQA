'''
this graph builder takes an arxiv id, and outputs the corresponding graph built from the paper
method: named entity recognition
alternatives: tfidf or cosine similarity
'''
import os
import re
import tarfile
import requests
import spacy
import itertools
from collections import defaultdict
import torch
from torch_geometric.data import Data
from hybrid_emb import get_hybrid_emb

def download_arxiv_latex(arxiv_id, save_dir):
    download_url = f"https://arxiv.org/e-print/{arxiv_id}"
    response = requests.get(download_url)

    if response.status_code == 200:
        tar_file_path = os.path.join(save_dir, f"{arxiv_id}.tar.gz")
        with open(tar_file_path, 'wb') as f:
            f.write(response.content)
        
        with tarfile.open(tar_file_path, 'r:gz') as tar:
            tar.extractall(path=save_dir)
        
        print(f"LaTeX files for {arxiv_id} extracted to {save_dir}")
    else:
        print(f"Failed to download LaTeX files for {arxiv_id}. Status code: {response.status_code}")

def build_graph_from_latex(tex_files):
    sections = []
    section_titles = []

    for tex_file in tex_files:
        with open(tex_file, 'r', encoding='utf-8') as file:
            content = file.read()

            # Use regex to split content into sections
            pattern = r'(\\section\{.*?\})'
            parts = re.split(pattern, content)

            for i in range(len(parts)):
                if parts[i].startswith('\\section'):
                    # Extract the title
                    section_title_match = re.match(r'\\section\{(.*?)\}', parts[i])
                    if section_title_match:
                        section_title = section_title_match.group(1)
                        section_titles.append(section_title)
                        if i + 1 < len(parts):
                            # The content immediately after the section title
                            section_content = parts[i + 1]
                            sections.append(section_content)
                        else:
                            sections.append('')
    return sections


nlp = spacy.load("en_core_web_sm")


def extract_entities(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    return entities


def compute_shared_entities(sections):
    section_entities = []
    edge_list = []
    edge_attr_list = []


    for idx, section in enumerate(sections):
        entities = extract_entities(section)
        section_entities.append(entities)


    for (i, entities_i), (j, entities_j) in itertools.combinations(enumerate(section_entities), 2):
        shared_entities = set(entities_i).intersection(set(entities_j)) 
        shared_count = len(shared_entities)
        if shared_count > 0:
            edge_list.append([i, j]) 
            edge_list.append([j, i]) 
            edge_attr_list.append(shared_count) 
            edge_attr_list.append(shared_count)

    return edge_list, edge_attr_list

def create_pytorch_graph(sections):
    edge_list, edge_attr_list = compute_shared_entities(sections)

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    num_nodes = len(sections)
    x = torch.arange(num_nodes, dtype=torch.float32).unsqueeze(1)

    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32).unsqueeze(1)

    graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return graph_data

def find_tex_files(directory):
    tex_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".tex"):
                tex_files.append(os.path.join(root, file))
    return tex_files

if __name__ == "__main__":
    arxiv_id = "2404.16130" 
    save_dir = "./latex_files"
    os.makedirs(save_dir, exist_ok=True)

    download_arxiv_latex(arxiv_id, save_dir)

    tex_files = find_tex_files(save_dir)

    sections = build_graph_from_latex(tex_files)

    graph = create_pytorch_graph(sections)

    print(f"Node Features (Sections):\n{graph.x}")
    print(f"Edge Index (References):\n{graph.edge_index}")
    print(f"Edge Attributes (Shared Entities):\n{graph.edge_attr}")
