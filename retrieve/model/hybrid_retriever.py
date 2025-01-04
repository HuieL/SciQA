import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import scipy.sparse as sp
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from src.model.CLEAR import ResidualDenseRetriever
import json


class ScoreFusionRetriever:
    # Method in https://arxiv.org/pdf/2010.01195
    def __init__(self, 
                 lambda_weight: float = 0.8,
                 specter2_model: str = "allenai/specter2_base",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize hybrid retriever combining sparse BOW and dense SPECTER2 embeddings.
        """
        self.lambda_weight = lambda_weight
        self.device = device

        # Initialize sparse components
        self.tfidf = TfidfVectorizer(stop_words='english')
        
        # Initialize dense components
        self.tokenizer = AutoTokenizer.from_pretrained(specter2_model)
        self.model = AutoModel.from_pretrained(specter2_model).to(device)
        
        # Storage for document embeddings
        self.doc_sparse_embeddings = None
        self.doc_dense_embeddings = None
        
    def _get_dense_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get dense embeddings using SPECTER2"""
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize and move to device
                inputs = self.tokenizer(text, 
                                      padding=True, 
                                      truncation=True,
                                      max_length=512,
                                      return_tensors="pt").to(self.device)
                
                # Get model output
                outputs = self.model(**inputs)
                
                # Use CLS token embedding
                embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
        
        return np.vstack(embeddings)
    
    def _get_sparse_embeddings(self, texts: List[str], fit: bool = False) -> sp.csr_matrix:
        """Get sparse TF-IDF embeddings"""
        if fit:
            return self.tfidf.fit_transform(texts)
        return self.tfidf.transform(texts)
    
    def index_documents(self, documents: List[str]):
        """Index documents to build sparse and dense representations"""
        # Get sparse embeddings
        self.doc_sparse_embeddings = self._get_sparse_embeddings(documents, fit=True)
        
        # Get dense embeddings
        self.doc_dense_embeddings = self._get_dense_embeddings(documents)
        
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between embeddings"""
        norm_a = np.linalg.norm(a, axis=1)
        norm_b = np.linalg.norm(b, axis=1)
        return np.dot(a, b.T) / np.outer(norm_a, norm_b)

    def _sparse_cosine_similarity(self, a: sp.csr_matrix, b: sp.csr_matrix) -> np.ndarray:
        """Compute cosine similarity between sparse matrices"""
        norm_a = np.sqrt(a.multiply(a).sum(axis=1))
        norm_b = np.sqrt(b.multiply(b).sum(axis=1))
        return np.array(a.dot(b.T).todense()) / np.outer(norm_a, norm_b)

    def search(self, 
              query: str, 
              top_k: int = 10) -> List[Dict[str, Any]]:
        # Get query embeddings
        query_sparse = self._get_sparse_embeddings([query])
        query_dense = self._get_dense_embeddings([query])
        
        # Calculate similarities
        sparse_scores = self._sparse_cosine_similarity(
            query_sparse, 
            self.doc_sparse_embeddings
        )[0]
        
        dense_scores = self._cosine_similarity(
            query_dense, 
            self.doc_dense_embeddings
        )[0]
        
        # Combine scores using weighted sum
        combined_scores = (
            self.lambda_weight * dense_scores + 
            (1 - self.lambda_weight) * sparse_scores
        )
        
        # Get top-k indices and scores
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        top_scores = combined_scores[top_indices]
        
        results = []
        for idx, score in zip(top_indices, top_scores):
            results.append({
                "index": int(idx),
                "score": float(score),
                "sparse_score": float(sparse_scores[idx]),
                "dense_score": float(dense_scores[idx])
            })
            
        return results

class ClearRetriever:
    # Method in https://arxiv.org/pdf/2004.13969
    def __init__(self, model_path: str, bert_model: str = "bert-base-uncased", device: str = "cuda"):
        self.device = device
        self.model = ResidualDenseRetriever(model_name=bert_model, device=device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

        # Storage for document embeddings
        self.doc_embeddings = None
        self.documents = []

    def _get_dense_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get dense embeddings for a list of texts using the CLEAR model."""
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), 64):  # Batch processing for efficiency
                batch = texts[i:i + 64]
                embeddings.append(self.model(batch).cpu().numpy())
        return np.vstack(embeddings)

    def index_documents(self, documents: List[str]):
        self.documents = documents
        self.doc_embeddings = self._get_dense_embeddings(documents)
        print(f"Indexed {len(documents)} documents.")

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        # Compute query embedding
        query_embedding = self._get_dense_embeddings([query])[0]  # Single query embedding

        # Compute cosine similarities
        norm_query = np.linalg.norm(query_embedding)
        norm_docs = np.linalg.norm(self.doc_embeddings, axis=1)
        similarities = np.dot(self.doc_embeddings, query_embedding) / (norm_docs * norm_query)

        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_scores = similarities[top_indices]

        results = []
        for idx, score in zip(top_indices, top_scores):
            results.append({
                "index": int(idx),
                "score": float(score),
                "document": self.documents[idx]
            })

        return results

class ColBERTRetriever:
    # Method in https://arxiv.org/pdf/2004.12832
    def __init__(self, 
                 model_name: str = "colbert-ir/colbertv2.0", 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 max_length: int = 512):
        """
        Initialize ColBERT retriever for hybrid retrieval.
        """
        self.device = device
        self.max_length = max_length

        # Initialize ColBERT tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)

        # Storage for token embeddings and documents
        self.doc_token_embeddings = []
        self.documents = []

    def _get_token_embeddings(self, texts: List[str]) -> List[torch.Tensor]:
        """Get token embeddings for a list of texts."""
        token_embeddings = []
        self.model.eval()
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length
                ).to(self.device)
                outputs = self.model(**inputs)
                token_embeddings.append(outputs.last_hidden_state.squeeze(0).cpu())
        return token_embeddings

    def index_documents(self, documents: List[str]):
        self.documents = documents
        self.doc_token_embeddings = self._get_token_embeddings(documents)

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve top-k documents for the given query."""
        query_token_embeddings = self._get_token_embeddings([query])[0]

        results = []
        for idx, doc_token_embedding in enumerate(self.doc_token_embeddings):
            similarity = torch.mm(query_token_embeddings, doc_token_embedding.T).max(dim=1).values.sum().item()
            results.append({"index": idx, "score": similarity, "document": self.documents[idx]})

        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

def load_from_json(json_file: str) -> (List[str], List[str]):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    questions = []
    contexts = []

    for key, value in data.items():
        questions.append(value['QUESTION'])
        combined_context = ' '.join(value['CONTEXTS'])
        contexts.append(combined_context)

    return questions, contexts

def test_hybrid_retrievers():
    # Sample documents
    queries, chunks = load_from_json("../../dataset/ori_pqal.json")
    # documents = [
    #     "The quick brown fox jumps over the lazy dog.",
    #     "Machine learning is a subset of artificial intelligence.",
    #     "Python is a popular programming language.",
    #     "Neural networks are inspired by biological neurons.",
    #     "Deep learning has revolutionized computer vision."
    # ]
    
    # query = "What is artificial intelligence?"
    
    # Initialize retrievers
    retrievers = {
        "ScoreFusion": ScoreFusionRetriever(lambda_weight=0.7),  # 70% dense, 30% sparse
        # "CLEAR": ClearRetriever(model_path="clear_model.pt", bert_model="bert-base-uncased"),
        "ColBERT": ColBERTRetriever()
    }
    
    # Index documents
    for name, retriever in retrievers.items():
        print(f"\nIndexing documents for {name}...")
        retriever.index_documents(chunks)
    
    # Test query
    for name, retriever in retrievers.items():
        print(f"\n{name} scores:")
        correct_count = 0
        match_count = 0
        for query_idx, query in enumerate(queries):
            print(query_idx)
            # print(f"\nQuery: '{query}'")
            # Use the `search` method for retrievers that support it
            results = retriever.search(query, top_k=3)
            print(query_idx)
            for rank, result in enumerate(results, 1):
                if result['index'] == query_idx:
                    match_count += 1
                if rank == 1 and result['index'] == query_idx:
                    correct_count += 1
                # print(f"{rank}. Score: {result['score']:.4f} - Document: {result['document']}")

        correct = correct_count / len(queries)
        print(f"\n CorrectAccuracy: {correct:.2%} ({correct_count}/{len(queries)})")
        match = match_count / len(queries)
        print(f"\n MatchAccuracy: {match:.2%} ({match_count}/{len(queries)})")

if __name__ == "__main__":
    test_hybrid_retrievers()
