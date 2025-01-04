import re
from collections import defaultdict
import torch
from torch_geometric.data import Data
from typing import List, Dict, Tuple


def paper2graph(title: str, tex_content: str, max_tokens: int = 1000) -> Data:
    """
    Convert LaTeX paper content to a graph representation.
    
    Args:
        tex_content (str): The LaTeX content of the paper
        max_tokens (int, optional): Maximum number of tokens per chunk. Defaults to 50.
    
    Returns:
        Data: PyTorch Geometric graph data object
    """
    sections = parse_sections(tex_content)
    node_list, edges, node_text = build_graph_from_sections(
        sections, 
        tex_content, 
        max_tokens=max_tokens,
        main_title=title
    )
    paper_graph = create_pytorch_graph(node_list, edges, node_text)

    section_labels = get_section_labels(tex_content)
    paper_graph.edge_index = build_reference_edges(paper_graph.node_text, paper_graph.edge_index, section_labels)

    return paper_graph


def split_into_chunks(text, max_tokens):
    words = text.split()
    chunks = [' '.join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]
    return chunks

def parse_sections(content):
    pattern = r'(\\(sub)*section\{(.*?)\})'
    sections = []
    
    # Find all section commands and their titles
    matches = list(re.finditer(pattern, content))
    for i, match in enumerate(matches):
        section_command = match.group(1) 
        title = match.group(3)  # Extract the section title
        start_index = match.start(1) 
        end_index = matches[i + 1].start(1) if i + 1 < len(matches) else len(content)
        
        # Determine if this is a leaf node by checking if the next section has a higher or same level
        is_leaf = True 
        if i + 1 < len(matches):
            next_command = matches[i + 1].group(1)
            # It's a leaf only if the next section is of a higher or equal hierarchy
            is_leaf = next_command.count("sub") <= section_command.count("sub")
        
        sections.append({
            "title": title,
            "start_index": start_index,
            "end_index": end_index,
            "is_leaf": is_leaf,
            "depth": section_command.count("sub")  # Depth indicates level of subsection
        })
    
    return sections

def build_graph_from_sections(sections, content, max_tokens, main_title="Main Title"):
    node_list = []
    edges = []
    node_text = []  
    section_refs = defaultdict(list)
    section_dict = {}  

    root_node = len(node_list)
    node_list.append({"title": main_title, "content": main_title, "type": "root"})
    node_text.append(main_title)  # Root node text attribute

    # Stack to maintain hierarchy of sections
    stack = [(root_node, -1)]  # (node_index, depth), with root node at depth -1

    for section in sections:
        section_node = len(node_list)
        node_list.append({"title": section["title"], "content": section["title"], "type": "section"})
        node_text.append(section["title"])  # Add section title to node_text
        section_dict[section["title"]] = section_node  # Map title to node index for cross-references

        # Pop items from the stack until we reach the correct parent level
        while stack and stack[-1][1] >= section["depth"]:
            stack.pop()

        parent_node = stack[-1][0]
        edges.append((parent_node, section_node))
        stack.append((section_node, section["depth"]))

        if section["is_leaf"]:
            section_content = content[section["start_index"]:section["end_index"]]
            chunks = split_into_chunks(section_content, max_tokens=max_tokens)

            for chunk in chunks:
                chunk_node = len(node_list)
                node_list.append({"title": "", "content": chunk, "type": "chunk"})
                node_text.append(chunk)
                edges.append((section_node, chunk_node))

                refs = re.findall(r'\\ref\{(.*?)\}', chunk)
                for ref in refs:
                    if ref in section_dict:
                        ref_node = section_dict[ref]
                        edges.append((chunk_node, ref_node))  # Add edge from chunk to referenced section

    return node_list, edges, node_text

def create_pytorch_graph(node_list, edges, node_text):
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    num_nodes = len(node_list)
    x = torch.arange(num_nodes, dtype=torch.float32).unsqueeze(1)

    graph_data = Data(x=x, edge_index=edge_index, node_text=node_text)

    return graph_data

def get_section_labels(content):
    # Pattern to match any section command and an optional label
    pattern = r'(\\(sub)*section\{(.*?)\})(?:\s*\\label\{(.*?)\})?'
    section_labels = {}

    # Find all section commands with optional labels
    matches = re.finditer(pattern, content)
    for match in matches:
        section_title = match.group(3)  # Extract the section title
        section_label = match.group(4)  # Extract the optional label, if present
        if section_label:
            section_labels[section_title] = section_label

    return section_labels

def build_reference_edges(node_texts: List[str], edge_index: torch.Tensor, section_labels: Dict[str, str]) -> torch.Tensor:
    label_to_node = {}
    
    for idx, text in enumerate(node_texts):
        section_pattern = r'\\(?:sub)*section\{(.*?)\}(?:\s*\\label\{(.*?)\})?'
        matches = re.search(section_pattern, text)
        if matches:
            section_title = matches.group(1)
            if section_title in section_labels:
                label = section_labels[section_title]
                label_to_node[label] = idx

    # Convert existing edges to list of tuples
    # try:
    #     existing_edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    # except:
    #     existing_edges = []

    existing_edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    all_edges = existing_edges.copy()
    
    # Second pass: find \ref{} commands and add edges
    ref_pattern = r'\\ref\{(.*?)\}'
    for source_idx, text in enumerate(node_texts):
        refs = re.finditer(ref_pattern, text)
        for ref in refs:
            label = ref.group(1)
            if label in label_to_node:
                target_idx = label_to_node[label]
                new_edge = (source_idx, target_idx)
                if new_edge not in all_edges:
                    all_edges.append(new_edge)

    if all_edges:
        source_nodes = [edge[0] for edge in all_edges]
        target_nodes = [edge[1] for edge in all_edges]
        new_edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    else:
        new_edge_index = edge_index.clone()
    
    return new_edge_index
