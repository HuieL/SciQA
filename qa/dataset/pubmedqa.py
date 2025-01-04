import os
import time

import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from typing import Dict, Union, Any, List
from torch_geometric.data import Data
import numpy as np
import scipy.sparse as sp
import argparse
import warnings
from src.model.dense_retriever import (
    MiniLMRetriever, SentenceBERTRetriever, LaBSERetriever,
    mContrieverRetriever, T5Retriever, E5Retriever, 
    BGEM3FlagRetriever, SPARRetriever
)
from src.model.sparse_retriever import (
    BM25Retriever, BGEM3Retriever,
    Doc2QueryRetriever, SPLADERetriever
)
from src.model.hybrid_retriever import (
    ScoreFusionRetriever, CLEARRetriever, ColBERTRetriever
)


warnings.filterwarnings("ignore")
data_path = './dataset/pubmed_qa'
cache_dir = 'dataset/pubmed_qa/cache/3_labels'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PubMedQADataset(Dataset):
    def __init__(self, args):
        super().__init__()

        graph_path = os.path.join(data_path, args.graph)
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Graph file not found at {graph_path}")
            
        self.graph = torch.load(graph_path)
        self.valid_indices = [idx for idx in range(len(self.graph.questions)) 
                            if self.graph.questions[idx]]
        
        # Load encoded graph
        cache_path = os.path.join(cache_dir, f'pubmed_{args.retriever}_encoded_graph.pt')
        if not os.path.exists(cache_path):
            raise ValueError(
                "Preprocessed data not found! Please first run: "
                "python -m src.dataset.pubmedqa "
                f"--retriever {args.retriever}"
            )
            
        self.encoded_graph = torch.load(cache_path)
        self.split_dict = self.get_idx_split()

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index: Union[int, str, slice]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        if isinstance(index, str):
            if index not in self.split_dict:
                raise KeyError(f"Split {index} not found. Available splits: {list(self.split_dict.keys())}")
            split_indices = self.split_dict[index]
            return [self[idx] for idx in split_indices]
            
        elif isinstance(index, slice):
            indices = range(*index.indices(len(self)))
            return [self[idx] for idx in indices]
            
        else:
            real_idx = self.valid_indices[index]
            return {
                'id': index,
                'real_id': real_idx,
                'question': self.encoded_graph.questions[real_idx],
                'label': self.encoded_graph.answers[real_idx].lower(),
                'retrieved_context': self.encoded_graph.retrieved_contexts[real_idx],
                'original_context': self.encoded_graph.contexts[real_idx]
            }

    def get_idx_split(self) -> Dict[str, List[int]]:
        """Get indices for train/val/test splits"""
        indices = list(range(len(self.valid_indices)))
        return {
            'train': [], 
            'val': [], 
            'test': indices
        }

    def get_split(self, split: str) -> List[Dict[str, Any]]:
        """Helper method to get all items in a split"""
        return self[split]

def encode_graph_with_retriever(graph: Data, retriever: Any, batch_size: int = 32) -> Data:
    texts = []
    for ctx in graph.contexts:
        if not ctx:
            texts.append("") 
        elif isinstance(ctx, list):
            texts.append(' '.join(str(c) for c in ctx))
        else:
            texts.append(str(ctx))
            
    encoded_graph = Data(
        edge_index=graph.edge_index,
        edge_attr=getattr(graph, 'edge_attr', None),
        contexts=graph.contexts,
        questions=graph.questions,
        answers=graph.answers,
        decisions=graph.decisions
    )

    print(f"Indexing documents with {type(retriever).__name__}...")
    try:
        valid_texts = [t for t in texts if t.strip()]
        retriever.index_documents(valid_texts, batch_size=batch_size)
    except Exception as e:
        print(f"Error during indexing: {str(e)}")
        print(f"Valid texts sample: {valid_texts[0] if valid_texts else 'No valid texts'}")
        raise
    
    try:
        if isinstance(retriever, (BM25Retriever, SPLADERetriever, Doc2QueryRetriever, BGEM3Retriever)):
            # For sparse retrievers
            encoded_graph.x_sparse = retriever.doc_embeddings
            encoded_graph.retriever_type = 'sparse'

        elif isinstance(retriever, (MiniLMRetriever, SentenceBERTRetriever, LaBSERetriever,
                                mContrieverRetriever, T5Retriever, E5Retriever,
                                BGEM3FlagRetriever, SPARRetriever)):
            # For dense retrievers
            encoded_graph.x_dense = retriever.doc_embeddings
            encoded_graph.retriever_type = 'dense'

        elif isinstance(retriever, ScoreFusionRetriever):
            # For ScoreFusion
            # Create empty arrays for texts that were empty
            dense_embeddings = []
            sparse_embeddings = []
            empty_dense = np.zeros(retriever.encode(["dummy text"])[0].shape[1])
            empty_sparse = sp.csr_matrix((1, retriever.tfidf.get_feature_names_out().shape[0]))

            for text in texts:
                if text.strip():
                    emb_dense, emb_sparse = retriever.encode([text])
                    dense_embeddings.append(emb_dense[0])
                    sparse_embeddings.append(emb_sparse)
                else:
                    dense_embeddings.append(empty_dense.copy())
                    sparse_embeddings.append(empty_sparse.copy())

            encoded_graph.x_dense = torch.tensor(np.stack(dense_embeddings))
            encoded_graph.x_sparse = sp.vstack(sparse_embeddings)
            encoded_graph.lambda_weight = retriever.lambda_weight
            encoded_graph.retriever_type = 'hybrid'

        elif isinstance(retriever, ColBERTRetriever):
            # For ColBERT
            encoded_graph.x_tokens = retriever.doc_embeddings
            encoded_graph.retriever_type = 'colbert'

        elif isinstance(retriever, CLEARRetriever):
            # For CLEAR
            encoded_graph.x_dense = retriever.doc_embeddings
            encoded_graph.retriever_type = 'clear'
        else:
            raise ValueError(f"Unknown retriever type: {type(retriever)}")

    except Exception as e:
        print(f"Error storing embeddings: {str(e)}")
        print(f"Retriever type: {type(retriever)}")
        print(f"Available attributes: {dir(retriever)}")
        raise

    return encoded_graph

def perform_retrieval(encoded_graph: Data, retriever: Any, 
                     top_k: int = 15, batch_size: int = 32) -> Data:
    retrieved_contexts = [""] * len(encoded_graph.contexts)
    valid_indices = [i for i, q in enumerate(encoded_graph.questions) if q]
    # print(f'这是索引{valid_indices}')
    try:
        for i in tqdm(range(len(valid_indices)), desc="Processing questions"):
            idx = valid_indices[i]
            question = str(encoded_graph.questions[idx])

            # 这里进行修改，经过debug发现，在search的过程中，会将空格作为分值最高的context，只留下非空的context进行search
            chunks = [" ".join(map(str, context)) for context in encoded_graph.contexts]
            chunks = [chunk for chunk in chunks if len(chunk) > 0]

            results = retriever.search(
                query=question,
                # documents=encoded_graph.contexts,
                documents=chunks,
                top_k=top_k,
                batch_size=batch_size
            )
            retrieved_docs = []
            for r in results:
                doc = r['document']
                if isinstance(doc, list):
                    retrieved_docs.append(' '.join(str(d) for d in doc))
                else:
                    retrieved_docs.append(str(doc))
            
            # Store joined retrieved documents at the correct index
            retrieved_contexts[idx] = ' '.join(retrieved_docs)

    except Exception as e:
        print(f"Error during retrieval: {str(e)}")
        print(f"Current question index: {i}")
        print(f"Number of processed questions: {sum(1 for c in retrieved_contexts if c)}")
        # print(f"Sample result structure: {results[0] if results else 'No results'}")
        raise

    encoded_graph.retrieved_contexts = retrieved_contexts
    return encoded_graph

def initialize_retrievers():
    return {
        # Dense retrievers
        'minilm': lambda: MiniLMRetriever(device),
        'sbert': lambda: SentenceBERTRetriever(device),
        'labse': lambda: LaBSERetriever(device),
        'mcontriever': lambda: mContrieverRetriever(device),
        't5': lambda: T5Retriever(device),
        'e5': lambda: E5Retriever(device),
        'bge_m3_flag': lambda: BGEM3FlagRetriever(device),
        'spar': lambda: SPARRetriever(device),
        
        # Sparse retrievers
        'bm25': lambda: BM25Retriever(),
        'bge_m3': lambda: BGEM3Retriever(device),
        'doc2query': lambda: Doc2QueryRetriever(device),
        'splade': lambda: SPLADERetriever(device),
        #
        # Hybrid retrievers
        'score_fusion': lambda: ScoreFusionRetriever(lambda_weight=0.7, device=device),
        'clear': lambda: CLEARRetriever(model_path="./clear_model.pt", device=device),
        'colbert': lambda: ColBERTRetriever(device)
    }

def preprocess_dataset(args):
    os.makedirs(cache_dir, exist_ok=True)
    cached_graph_path = os.path.join(cache_dir, f'pubmed_{args.retriever}_encoded_graph.pt')
    
    if os.path.exists(cached_graph_path):
        print(f"Found cached encoded graph for {args.retriever}")
        return
        
    print(f"No cached graph found. Preprocessing dataset with {args.retriever}...")
    
    # Load graph
    graph_path = os.path.join(data_path, args.graph)
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph file not found at {graph_path}")
    graph = torch.load(graph_path)
    
    # Initialize retriever
    retrievers = initialize_retrievers()
    if args.retriever not in retrievers:
        raise ValueError(f"Unknown retriever: {args.retriever}")
        
    retriever = retrievers[args.retriever]()
    
    try:
        # Encode graph
        encoded_graph = encode_graph_with_retriever(graph, retriever, args.batch_size)
        print("Graph encoding completed")
        
        # Perform retrieval
        encoded_graph = perform_retrieval(
            encoded_graph, 
            retriever,
            top_k=args.topk,
            batch_size=args.batch_size
        )
        print("Retrieval completed")
        
        print(f"Saving encoded graph to {cached_graph_path}")
        torch.save(encoded_graph, cached_graph_path)
        
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        print("Full error details:", e)
        raise

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess PubMedQA dataset')
    parser.add_argument('--graph', type=str, default='new_pqal_graph.pt',
                       help='Path to citation graph')
    parser.add_argument('--retriever', type=str, default='bge_m3',
                       choices=list(initialize_retrievers().keys()),
                       help='Type of retriever to use')
    parser.add_argument('--topk', type=int, default=1,
                       help='Number of contexts to retrieve (default: 15)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for processing (default: 32)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    preprocess_dataset(args)
    dataset = PubMedQADataset(args)
