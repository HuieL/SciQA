import torch 
from sentence_transformers import SentenceTransformer
from transformers import (
    T5Model, T5Tokenizer, 
    AutoModel, AutoTokenizer,
    PreTrainedModel, PreTrainedTokenizer
)
from typing import List, Union, Dict, Any
import numpy as np
from FlagEmbedding import BGEM3FlagModel


class BaseRetriever:
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.tokenizer = None
    
    def encode(self, texts: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def compute_similarity(self, query_embeddings: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and document embeddings"""
        return np.dot(query_embeddings, doc_embeddings.T)

    def search(self, query: str, documents: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """Basic search functionality"""
        query_embedding = self.encode([query])
        doc_embeddings = self.encode(documents)
        
        scores = self.compute_similarity(query_embedding, doc_embeddings)[0]
        top_idx = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_idx:
            results.append({
                'document': documents[idx],
                'score': float(scores[idx]),
                'index': int(idx)
            })
        return results

class MiniLMRetriever(BaseRetriever):
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(device)
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2').to(device)
    
    def encode(self, texts: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        return self.model.encode(texts, batch_size=batch_size, device=self.device, **kwargs)

class SentenceBERTRetriever(BaseRetriever):
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(device)
        self.model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens').to(device)
    
    def encode(self, texts: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        return self.model.encode(texts, batch_size=batch_size, device=self.device, **kwargs)

class LaBSERetriever(BaseRetriever):
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(device)
        self.model = SentenceTransformer('sentence-transformers/LaBSE').to(device)
    
    def encode(self, texts: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        return self.model.encode(texts, batch_size=batch_size, device=self.device, **kwargs)

class mContrieverRetriever(BaseRetriever):
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(device)
        self.model = AutoModel.from_pretrained('facebook/mcontriever-msmarco').to(device)
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/mcontriever-msmarco')
        
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def encode(self, texts: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded_input = self.tokenizer(batch_texts, padding=True, truncation=True, 
                                        max_length=512, return_tensors='pt').to(self.device)
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            all_embeddings.append(sentence_embeddings.cpu().numpy())
        return np.vstack(all_embeddings)

class T5Retriever(BaseRetriever):
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(device)
        self.model = T5Model.from_pretrained('t5-base').to(device)
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        
    def encode(self, texts: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded_input = self.tokenizer(batch_texts, padding=True, truncation=True, 
                                        max_length=512, return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = self.model.encoder(**encoded_input)
                # Use [CLS] token embedding or mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1)
            all_embeddings.append(embeddings.cpu().numpy())
        return np.vstack(all_embeddings)

class E5Retriever(BaseRetriever):
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(device)
        self.model = AutoModel.from_pretrained('intfloat/e5-base').to(device)
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base')
    
    def encode(self, texts: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            # Add prefix for queries: "query: "
            encoded_input = self.tokenizer(batch_texts, padding=True, truncation=True,
                                        max_length=512, return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = self.model(**encoded_input)
                embeddings = outputs.last_hidden_state[:, 0]  # Use [CLS] token
            all_embeddings.append(embeddings.cpu().numpy())
        return np.vstack(all_embeddings)

class BGEM3FlagRetriever(BaseRetriever):
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(device)
        self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    
    def encode(self, texts: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        outputs = self.model.encode(
            texts, batch_size=batch_size, max_length=8192, **kwargs
        )
        return outputs['dense_vecs']
    
class SPARRetriever(BaseRetriever):
    def __init__(self, model_name="facebook/spar-paq-bm25-lexmodel-context-encoder", device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(device)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def encode(self, texts: List[str], batch_size=32, **kwargs) -> np.ndarray:
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                inputs = self.tokenizer(batch, padding=True, truncation=True, 
                                        max_length=512, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]  # Use CLS token
                embeddings.append(cls_embeddings.cpu().numpy())
        return np.vstack(embeddings)


# Example usage
# def test_retrievers():
#     # Sample documents
#     documents = [
#         "The quick brown fox jumps over the lazy dog.",
#         "Machine learning is a subset of artificial intelligence.",
#         "Python is a popular programming language.",
#         "Neural networks are inspired by biological neurons.",
#         "Deep learning has revolutionized computer vision."
#     ]
    
#     query = "What is artificial intelligence?"
    
#     # Initialize retrievers
#     retrievers = {
#         "MiniLM": MiniLMRetriever(),
#         "SentenceBERT": SentenceBERTRetriever(),
#         "LaBSE": LaBSERetriever(),
#         "mContriever": mContrieverRetriever(),
#         "T5": T5Retriever(),
#         "E5": E5Retriever(),
#         "BGE-M3-Flag": BGEM3FlagRetriever(),
#         "SPAR": SPARRetriever()
#     }
    
#     # Test each retriever
#     for name, retriever in retrievers.items():
#         print(f"\nResults for {name}:")
#         results = retriever.search(query, documents, top_k=3)
#         for rank, result in enumerate(results, 1):
#             print(f"{rank}. Score: {result['score']:.4f} - {result['document']}")

# if __name__ == "__main__":
#     test_retrievers()
