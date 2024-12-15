import torch
from torch_geometric.data import Data
from src.utils.paper2graph import paper2graph


def to_hierarchical(citation_graph: Data, chunk_length: int) -> Data:
    """
    Convert a citation graph to a hierarchical version.
            
    Returns:
        Data: hierarchical citation graph
    """
    list_of_paper_graphs = []
    for paper_index in range(len(citation_graph.title)):
        try:
            list_of_paper_graphs.append(paper2graph(citation_graph.title[paper_index], citation_graph.content[paper_index], chunk_length))
        except:
            continue

    merged_graph = merge_paper_graphs(list_of_paper_graphs)
    hier_graph = hier_graphs(merged_graph, citation_graph)

    return hier_graph

def merge_paper_graphs(paper_graphs):
    total_nodes = 0
    all_edge_index = [] 
    all_node_text = [] 
    root_labels = [] 

    for graph in paper_graphs:
        num_nodes = graph.x.size(0)

        adjusted_edge_index = graph.edge_index + total_nodes

        all_edge_index.append(adjusted_edge_index)
        all_node_text.extend(graph.node_text)

        root_label = torch.zeros(num_nodes, dtype=torch.long)
        root_label[0] = 1  # Assuming the root node has index 0
        root_labels.append(root_label)

        total_nodes += num_nodes

    # Concatenate all data to form the merged graph
    merged_edge_index = torch.cat(all_edge_index, dim=1)
    merged_node_text = all_node_text
    merged_root_labels = torch.cat(root_labels, dim=0)
    merged_graph = Data(edge_index=merged_edge_index, node_text=merged_node_text, root_labels=merged_root_labels)
    
    return merged_graph

def hier_graphs(merged_paper_graph, citation_graph):
    root_indices = torch.where(merged_paper_graph.root_labels == 1)[0]

    chunks_to_idx = {merged_paper_graph.node_text[idx]: idx for idx in root_indices}
    cit_title_to_idx = {title: idx for idx, title in enumerate(citation_graph.title)}
    cit_idx_to_title = {idx: title for idx, title in enumerate(citation_graph.title)}

    new_edges = []

    for hier_idx in root_indices:
        hier_title = merged_paper_graph.node_text[hier_idx]

        if hier_title in cit_title_to_idx:
            cit_idx = cit_title_to_idx[hier_title]

            # Get all citations from this paper in citation graph
            citation_edges = citation_graph.edge_index
            source_mask = citation_edges[0] == cit_idx
            cited_indices = citation_edges[1][source_mask]

            for cited_idx in cited_indices:
                cited_title = cit_idx_to_title[cited_idx.item()]
                
                if cited_title in chunks_to_idx:
                    cited_hier_idx = chunks_to_idx[cited_title]
                    new_edges.append((hier_idx, cited_hier_idx))

            # Also get papers that cite this paper
            target_mask = citation_edges[1] == cit_idx
            citing_indices = citation_edges[0][target_mask]
            
            for citing_idx in citing_indices:
                citing_title = cit_idx_to_title[citing_idx.item()]
                
                if citing_title in chunks_to_idx:
                    citing_hier_idx = chunks_to_idx[citing_title]
                    new_edges.append((citing_hier_idx, hier_idx))

    if new_edges:
        new_edges_tensor = torch.tensor(new_edges, dtype=torch.long).t()
        updated_edge_index = torch.cat([merged_paper_graph.edge_index, new_edges_tensor], dim=1)

        updated_graph = Data(
            edge_index=updated_edge_index,
            node_text=merged_paper_graph.node_text,
            root_labels=merged_paper_graph.root_labels
        )
    else:
        updated_graph = merged_paper_graph
    
    return updated_graph