import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, T5Model, T5Tokenizer
from typing import List
from src.model.base_retriever import DenseRetriever


class MiniLMRetriever(DenseRetriever):
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(device)
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2').to(device)

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        return self.model.encode(texts, batch_size=batch_size, device=self.device)

class SentenceBERTRetriever(DenseRetriever):
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(device)
        self.model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens').to(device)

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        return self.model.encode(texts, batch_size=batch_size, device=self.device)

class LaBSERetriever(DenseRetriever):
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(device)
        self.model = SentenceTransformer('sentence-transformers/LaBSE').to(device)

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        return self.model.encode(texts, batch_size=batch_size, device=self.device)

class mContrieverRetriever(DenseRetriever):
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(device)
        self.model = AutoModel.from_pretrained('facebook/mcontriever-msmarco').to(device)
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/mcontriever-msmarco')

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True,
                                  max_length=512, return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings.append(outputs.last_hidden_state[:, 0].cpu().numpy())
        return np.vstack(embeddings)

class T5Retriever(DenseRetriever):
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(device)
        self.model = T5Model.from_pretrained('t5-base').to(device)
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True,
                                  max_length=512, return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = self.model.encoder(**inputs)
                embeddings.append(outputs.last_hidden_state.mean(dim=1).cpu().numpy())
        return np.vstack(embeddings)

class E5Retriever(DenseRetriever):
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(device)
        self.model = AutoModel.from_pretrained('intfloat/e5-base').to(device)
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base')

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True,
                                  max_length=512, return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings.append(outputs.last_hidden_state[:, 0].cpu().numpy())
        return np.vstack(embeddings)

class BGEM3FlagRetriever(DenseRetriever):
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(device)
        from FlagEmbedding import BGEM3FlagModel
        self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        outputs = self.model.encode(texts, batch_size=batch_size)
        return outputs['dense_vecs']

class SPARRetriever(DenseRetriever):
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(device)
        self.model = AutoModel.from_pretrained(
            'facebook/spar-paq-bm25-lexmodel-context-encoder').to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            'facebook/spar-paq-bm25-lexmodel-context-encoder')

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True,
                                  max_length=512, return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings.append(outputs.last_hidden_state[:, 0].cpu().numpy())
        return np.vstack(embeddings)
    

