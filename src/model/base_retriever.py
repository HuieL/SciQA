from abc import ABC, abstractmethod
import torch
import numpy as np
from typing import List, Dict, Any, Union
import scipy.sparse as sp

class BaseRetriever(ABC):
    """Base class for all retrievers"""
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.tokenizer = None
        self.is_sparse = False
        
    @abstractmethod
    def encode(self, texts: List[str], batch_size: int = 32) -> Union[np.ndarray, sp.csr_matrix]:
        raise NotImplementedError
        
    def index_documents(self, documents: List[str], batch_size: int = 32):
        self.documents = documents
        self.doc_embeddings = self.encode(documents, batch_size)
        
    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def compute_similarity(self, 
                         query_vectors: Union[np.ndarray, sp.csr_matrix],
                         doc_vectors: Union[np.ndarray, sp.csr_matrix]) -> np.ndarray:
        if isinstance(query_vectors, np.ndarray) and isinstance(doc_vectors, np.ndarray):
            if len(query_vectors.shape) == 1:
                query_vectors = query_vectors.reshape(1, -1)
            # Normalize for cosine similarity
            query_norm = np.linalg.norm(query_vectors, axis=1, keepdims=True)
            doc_norm = np.linalg.norm(doc_vectors, axis=1, keepdims=True)
            query_normalized = query_vectors / (query_norm + 1e-8)  # Add epsilon to avoid division by zero
            doc_normalized = doc_vectors / (doc_norm + 1e-8)
            return np.dot(query_normalized, doc_normalized.T)
            
        elif isinstance(query_vectors, sp.csr_matrix) and isinstance(doc_vectors, sp.csr_matrix):
            similarity = query_vectors.dot(doc_vectors.T)
            if isinstance(similarity, sp.spmatrix):
                return similarity.toarray()
            return similarity
        else:
            raise ValueError("Query and document vectors must both be either dense or sparse")

    def search(self, query: Union[str, List[str]], documents: List[str], 
              top_k: int = 3, batch_size: int = 32) -> List[Dict[str, Any]]:
        if not hasattr(self, 'doc_embeddings'):
            self.index_documents(documents, batch_size)
            
        # Handle single query or list of queries
        if isinstance(query, str):
            queries = [query]
        else:
            queries = query
        
        query_vectors = self.encode(queries, batch_size)
        similarities = self.compute_similarity(query_vectors, self.doc_embeddings)
        
        if torch.is_tensor(similarities):
            similarities = similarities.cpu().numpy()
        if len(queries) == 1:
            similarities = similarities.reshape(1, -1)
            
        # Get results for each query
        all_results = []
        for i in range(similarities.shape[0]):
            scores = similarities[i]
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])][::-1]
            
            results = []
            for idx in top_indices:
                results.append({
                    'document': documents[int(idx)], 
                    'score': float(scores[idx]),
                    'index': int(idx)
                })
            all_results.append(results)
    
        return all_results[0] if isinstance(query, str) else all_results

class DenseRetriever(BaseRetriever):
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(device)
        self.is_sparse = False

class SparseRetriever(BaseRetriever):
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(device)
        self.is_sparse = True

class HybridRetriever(BaseRetriever):
    """Base class for hybrid retrievers"""
    def __init__(self, 
                 lambda_weight: float = 0.5,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(device)
        self.lambda_weight = lambda_weight  # Weight between dense and sparse scores
    
    def encode(self, texts: List[str], batch_size: int = 32) -> tuple:
        raise NotImplementedError
        
    def search(self, query: str, documents: List[str], top_k: int = 3, 
              batch_size: int = 32) -> List[Dict[str, Any]]:
        if not hasattr(self, 'doc_embeddings'):
            self.index_documents(documents, batch_size)
            
        # Get both dense and sparse query vectors
        query_dense, query_sparse = self.encode([query], batch_size)
        doc_dense, doc_sparse = self.doc_embeddings
        
        dense_scores = self.compute_similarity(query_dense, doc_dense)[0]
        sparse_scores = self.compute_similarity(query_sparse, doc_sparse)[0]
        combined_scores = (self.lambda_weight * dense_scores + 
                         (1 - self.lambda_weight) * sparse_scores)
        
        top_indices = np.argsort(combined_scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'document': documents[idx],
                'score': float(combined_scores[idx]),
                'dense_score': float(dense_scores[idx]),
                'sparse_score': float(sparse_scores[idx]),
                'index': int(idx)
            })
        return results
