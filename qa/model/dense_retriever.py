import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, T5Model, T5Tokenizer
from typing import List
from src.model.base_retriever import DenseRetriever
import json


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


def test_retrievers():
    # Sample documents
    chunks = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language.",
        "Neural networks are inspired by biological neurons.",
        "Deep learning has revolutionized computer vision."
    ]

    queries = ["What is brown fox",
               "What is artificial intelligence?",
               "What is Python",
               "What is neural networks",
               "What is deep learning",
    ]

    # Initialize retrievers
    retrievers = {
        "MiniLM": MiniLMRetriever(),
        "SentenceBERT": SentenceBERTRetriever(),
        "LaBSE": LaBSERetriever(),
        "mContriever": mContrieverRetriever(),
        "T5": T5Retriever(),
        "E5": E5Retriever(),
        "BGE-M3-Flag": BGEM3FlagRetriever(),
        "SPAR": SPARRetriever()
    }

    # 用于存储结果
    results_to_save = []

    for name, retriever in retrievers.items():
        print(f"\n{name} scores:")
        correct_count = 0
        match_count = 0
        for query_idx, query in enumerate(queries):
            if query_idx % 10 == 0:
                print(query_idx)
            results = retriever.search(query, chunks, top_k=3)

            # 保存每个retriever的结果
            result_entry = {
                "retriever_name": name,
                "query": query,
                "query_idx": query_idx,
                "results": [
                    result['document']  # 仅保留文档内容
                    for result in results
                ]
            }
            results_to_save.append(result_entry)

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

    # 将结果保存到文件
    with open("../dataset/dense_retriever_results.json", "w") as f:
        json.dump(results_to_save, f, indent=4)
    print("\nResults saved to 'retriever_results.json'")


if __name__ == "__main__":
    test_retrievers()


