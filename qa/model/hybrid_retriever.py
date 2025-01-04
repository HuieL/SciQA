import torch
import numpy as np
import os
import scipy.sparse as sp
from typing import List, Tuple, Dict, Any
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from src.model.CLEAR import ResidualDenseRetriever
from src.model.base_retriever import HybridRetriever


class ScoreFusionRetriever(HybridRetriever):
    # Method in https://arxiv.org/pdf/2010.01195
    def __init__(self, lambda_weight: float = 0.8, 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(lambda_weight=lambda_weight, device=device)
        # Initialize sparse components
        self.tfidf = TfidfVectorizer(stop_words='english')
        # Initialize dense components
        self.model = AutoModel.from_pretrained('allenai/specter2_base').to(device)
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')

    def encode(self, texts: List[str], batch_size: int = 32) -> Tuple[np.ndarray, sp.csr_matrix]:
        # Get dense embeddings
        dense_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True,
                                  max_length=512, return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                dense_embeddings.append(outputs.last_hidden_state[:, 0].cpu().numpy())
        dense_vectors = np.vstack(dense_embeddings)

        # Get sparse embeddings
        if len(texts) > 1 and not hasattr(self, 'doc_embeddings'): 
            sparse_vectors = self.tfidf.fit_transform(texts)
        else: 
            sparse_vectors = self.tfidf.transform(texts)

        return dense_vectors, sparse_vectors

class ColBERTRetriever(HybridRetriever):
    # Method in https://arxiv.org/pdf/2004.12832
    def __init__(self, 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 max_length: int = 512):
        super().__init__(device=device)
        self.max_length = max_length
        self.model = AutoModel.from_pretrained('colbert-ir/colbertv2.0').to(device)
        self.tokenizer = AutoTokenizer.from_pretrained('colbert-ir/colbertv2.0')

    def encode(self, texts: List[str], batch_size: int = 32) -> List[torch.Tensor]:
        token_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                token_embeddings.extend(outputs.last_hidden_state)

        return token_embeddings

    def search(self, query: str, documents: List[str], top_k: int = 3, 
              batch_size: int = 32) -> List[Dict[str, Any]]:
        # Index documents if not already done
        if not hasattr(self, 'doc_embeddings'):
            self.index_documents(documents, batch_size)

        query_tokens = self.encode([query], batch_size)[0]
        scores = []
        for doc_tokens in self.doc_embeddings:
            sim_matrix = torch.matmul(query_tokens, doc_tokens.T)
            score = torch.sum(torch.max(sim_matrix, dim=1)[0]).item()
            scores.append(score)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'document': documents[idx],
                'score': float(scores[idx]),
                'index': int(idx)
            })
        
        return results

class CLEARRetriever(HybridRetriever):
    # Method in https://arxiv.org/pdf/2004.13969
    def __init__(self, 
                 model_path: str, 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 bert_model: str = "bert-base-uncased"
                 ):
        super().__init__(device=device)

        # Check if model checkpoint exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"CLEAR model checkpoint not found at: {model_path}\n"
                f"The model should be trained first: ."
            )
            
        self.model = ResidualDenseRetriever(model_name=bert_model, device=device)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            raise RuntimeError(
                f"Error loading CLEAR model checkpoint: {str(e)}\n"
                "Make sure the checkpoint is compatible with the BERT model version."
            )
            
        self.model.eval()
        self.doc_embeddings = None
        self.documents = []

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                embeddings.append(self.model(batch).cpu().numpy())
        return np.vstack(embeddings)

    def index_documents(self, documents: List[str], batch_size: int = 64):
        self.documents = documents
        self.doc_embeddings = self.encode(documents, batch_size)

    def search(self, query: str, documents: List[str], top_k: int = 3, 
              batch_size: int = 64) -> List[Dict[str, Any]]:
        if not hasattr(self, 'doc_embeddings'):
            self.index_documents(documents, batch_size)

        query_embedding = self.encode([query], batch_size)[0] 
        norm_query = np.linalg.norm(query_embedding)
        norm_docs = np.linalg.norm(self.doc_embeddings, axis=1)
        similarities = np.dot(self.doc_embeddings, query_embedding) / (norm_docs * norm_query)
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_scores = similarities[top_indices]
        
        results = []
        for idx, score in zip(top_indices, top_scores):
            results.append({
                'document': self.documents[idx],
                'score': float(score),
                'index': int(idx)
            })
        
        return results
