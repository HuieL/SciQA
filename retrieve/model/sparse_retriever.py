from collections import Counter
import math
import numpy as np
from FlagEmbedding import BGEM3FlagModel
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModel
from typing import List, Tuple, Dict, Any
import scipy.sparse as sp
import json


class BaseRetriever:
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization by splitting on whitespace and converting to lowercase"""
        return text.lower().split()

    def score(self, query: str) -> List[Tuple[int, float]]:
        raise NotImplementedError

class BM25(BaseRetriever):
    def __init__(self, chunks: List[str], k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 with document chunks and parameters
        """
        self.k1 = k1
        self.b = b
        self.chunks = chunks
        self.N = len(chunks)
        
        # Calculate document frequencies and lengths
        self.doc_freqs = {}
        self.doc_lengths = []
        total_len = 0
        
        for chunk in chunks:
            terms = self._tokenize(chunk)
            length = len(terms)
            self.doc_lengths.append(length)
            total_len += length
            
            # Update document frequencies
            for term in set(terms):
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1
        
        self.avgdl = total_len / self.N if self.N > 0 else 0

    def score(self, query: str) -> List[Tuple[int, float]]:
        query_terms = self._tokenize(query)
        scores = []
        
        for idx, chunk in enumerate(self.chunks):
            score = 0
            chunk_terms = self._tokenize(chunk)
            term_freqs = Counter(chunk_terms)
            doc_len = self.doc_lengths[idx]
            
            for term in query_terms:
                if term not in self.doc_freqs:
                    continue
                
                # Calculate IDF
                idf = math.log((self.N - self.doc_freqs[term] + 0.5) / 
                             (self.doc_freqs[term] + 0.5) + 1)
                
                # Calculate normalized term frequency
                tf = term_freqs[term]
                norm_tf = ((tf * (self.k1 + 1)) / 
                          (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
                
                score += idf * norm_tf
            
            scores.append((idx, score))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)

class TFIDF(BaseRetriever):
    def __init__(self, chunks: List[str]):
        """Initialize TF-IDF with document chunks"""
        self.chunks = chunks
        self.N = len(chunks)
        
        # Calculate document frequencies
        self.doc_freqs = {}
        for chunk in chunks:
            terms = set(self._tokenize(chunk))
            for term in terms:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

    def score(self, query: str) -> List[Tuple[int, float]]:
        query_terms = self._tokenize(query)
        scores = []
        
        for idx, chunk in enumerate(self.chunks):
            score = 0
            chunk_terms = self._tokenize(chunk)
            term_freqs = Counter(chunk_terms)
            
            for term in query_terms:
                if term not in self.doc_freqs:
                    continue
                
                # Calculate TF-IDF
                tf = term_freqs[term]
                idf = math.log(self.N / self.doc_freqs[term] + 1)
                score += tf * idf
            
            scores.append((idx, score))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)

class BooleanRetriever(BaseRetriever):
    def __init__(self, chunks: List[str]):
        """Initialize Boolean Retriever with document chunks"""
        self.chunks = chunks
        self.chunk_terms = [set(self._tokenize(chunk)) for chunk in chunks]

    def score(self, query: str) -> List[Tuple[int, float]]:
        query_terms = set(self._tokenize(query))
        scores = []
        
        for idx, chunk_terms in enumerate(self.chunk_terms):
            # Coordination factor: ratio of matching terms
            matching_terms = len(query_terms & chunk_terms)
            score = matching_terms / len(query_terms) if query_terms else 0
            scores.append((idx, score))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)

class ExtendedBoolean(BaseRetriever):
    def __init__(self, chunks: List[str], p: float = 2.0):
        """
        Initialize Extended Boolean Retriever with document chunks
        p: p-norm parameter (typically 2.0 for Euclidean norm)
        """
        self.chunks = chunks
        self.p = p
        self.N = len(chunks)
        
        # Calculate IDF weights for term importance
        self.doc_freqs = {}
        for chunk in chunks:
            terms = set(self._tokenize(chunk))
            for term in terms:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

    def score(self, query: str) -> List[Tuple[int, float]]:
        query_terms = self._tokenize(query)
        scores = []
        
        for idx, chunk in enumerate(self.chunks):
            chunk_terms = self._tokenize(chunk)
            term_freqs = Counter(chunk_terms)
            
            # Calculate weighted p-norm similarity
            sum_weights = 0
            for term in query_terms:
                if term not in self.doc_freqs:
                    continue
                
                # Calculate term weight using TF-IDF
                tf = term_freqs[term]
                idf = math.log(self.N / self.doc_freqs[term] + 1)
                weight = tf * idf
                sum_weights += weight ** self.p
            
            # Final score using p-norm
            score = (sum_weights / len(query_terms)) ** (1/self.p) if query_terms else 0
            scores.append((idx, score))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)

class BGEM3Retriever(BaseRetriever):
    def __init__(self, chunks: List[str], model_name: str = 'BAAI/bge-m3', use_fp16: bool = True):
        """
        Initialize BGE-M3 retriever with document chunks and model.
        """
        self.chunks = chunks
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        
        # Precompute lexical weights for chunks
        self.chunk_lexical_weights = [
            self.model.encode([chunk], return_dense=False, return_sparse=True)['lexical_weights'][0]
            for chunk in chunks
        ]

    def score(self, query: str) -> List[Tuple[int, float]]:
        query_lexical_weights = self.model.encode([query], return_dense=False, return_sparse=True)['lexical_weights'][0]
        scores = []

        for idx, chunk_weights in enumerate(self.chunk_lexical_weights):
            # Use lexical matching score from BGE-M3
            score = self.model.compute_lexical_matching_score(query_lexical_weights, chunk_weights)
            scores.append((idx, score))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)

class Doc2QueryRetriever(BaseRetriever):
    def __init__(self, 
                 chunks: List[str], 
                 model_name: str = "doc2query/msmarco-t5-base-v1",
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 num_queries: int = 3):
        """
        Initialize the Doc2Query retriever.
        """
        self.chunks = chunks
        self.device = device
        self.num_queries = num_queries
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        
        # Expand documents with synthetic queries
        print("Generating synthetic queries for all document chunks...")
        self.expanded_chunks = [self._expand_document(chunk) for chunk in chunks]

    def _generate_queries(self, document: str) -> List[str]:
        inputs = self.tokenizer.encode(
            document,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs, 
                max_length=64,  # Max length for synthetic queries
                num_return_sequences=self.num_queries,
                do_sample=True,  # Sampling for diverse queries
                top_k=50,
                top_p=0.95
            )
        
        queries = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return queries

    def _expand_document(self, document: str) -> str:
        synthetic_queries = self._generate_queries(document)
        expanded_document = document + " " + " ".join(synthetic_queries)
        return expanded_document

    def score(self, query: str) -> List[Tuple[int, float]]:
        retriever = BM25(self.expanded_chunks)  # Use BM25 with expanded chunks
        return retriever.score(query)

class SPLADERetriever(BaseRetriever):
    def __init__(self, 
                 model_name: str = "naver/splade-cocondenser-ensembledistil", 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device = device

        # Initialize SPLADE tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)

        # Storage for sparse document representations
        self.doc_sparse_embeddings = None

    def _get_sparse_embeddings(self, texts: List[str]) -> sp.csr_matrix:
        """Generate sparse embeddings for a list of texts."""
        sparse_embeddings = []

        self.model.eval()
        with torch.no_grad():
            for text in texts:
                # Tokenize input
                inputs = self.tokenizer(
                    text,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)

                # Generate sparse logits
                outputs = self.model(**inputs)
                logits = outputs.last_hidden_state.sum(dim=1)  # Sum over token dimension

                # Convert logits to sparse representation
                sparse_vector = logits.squeeze(0).cpu().numpy()
                sparse_embeddings.append(sparse_vector)

        # Convert to CSR sparse matrix
        return sp.csr_matrix(sparse_embeddings)

    def index_documents(self, documents: List[str]):
        """Index documents to create their sparse embeddings."""
        self.documents = documents
        self.doc_sparse_embeddings = self._get_sparse_embeddings(documents)
        print(f"Indexed {len(documents)} documents with SPLADE++.")

    def _sparse_cosine_similarity(self, query: sp.csr_matrix, documents: sp.csr_matrix) -> np.ndarray:
        """Compute cosine similarity between query and document sparse embeddings."""
        norm_query = np.sqrt(query.multiply(query).sum(axis=1))
        norm_docs = np.sqrt(documents.multiply(documents).sum(axis=1))
        similarity = np.array(query.dot(documents.T).todense()) / np.outer(norm_query, norm_docs)
        return similarity.squeeze()

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve top-k documents for the given query."""
        # Get sparse embedding for the query
        query_sparse = self._get_sparse_embeddings([query])

        # Calculate similarities
        scores = self._sparse_cosine_similarity(query_sparse, self.doc_sparse_embeddings)

        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[top_indices]

        return [
            {"index": int(idx), "score": float(score), "document": self.documents[idx]}
            for idx, score in zip(top_indices, top_scores)
        ]


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
# Example usage
def test_retrievers():
    queries, chunks = load_from_json("../../dataset/ori_pqal.json")

    # # Sample documents
    # chunks = [
    #     "the quick brown fox jumps over the lazy dog",
    #     "a quick brown cat sleeps on the windowsill",
    #     "the lazy dog barks at the mailman",
    #     "a fox and a dog play in the garden"
    # ]

    # Initialize retrievers
    retrievers = {
        # "BM25": BM25(chunks),
        # "TF-IDF": TFIDF(chunks),
        # "Boolean": BooleanRetriever(chunks),
        # "Extended Boolean": ExtendedBoolean(chunks),
        # "BGE-M3 Sparse Retriever": BGEM3Retriever(chunks),
        # "Doc2Query": Doc2QueryRetriever(chunks),
        "SPLADE++": SPLADERetriever()
    }

    # Index documents for retrievers that require indexing
    for name, retriever in retrievers.items():
        if hasattr(retriever, "index_documents"):
            print(f"\nIndexing documents for {name}...")
            retriever.index_documents(chunks)

    # # Test queries
    # queries = [
    #     "quick brown",
    #     "lazy dog",
    #     "fox garden"
    # ]

    # print("Document chunks:")
    # for idx, chunk in enumerate(chunks):
    #     print(f"{idx}: {chunk}")


    for name, retriever in retrievers.items():
        print(f"\n{name} scores:")
        correct_count = 0
        match_count = 0
        for query_idx, query in enumerate(queries):
            if query_idx % 10 == 0:
                print(query_idx)
            # print(f"\nQuery: '{query}'")
            if hasattr(retriever, "search"):
                # Use the `search` method for retrievers that support it
                results = retriever.search(query, top_k=3)
                for rank, result in enumerate(results, 1):
                    if result['index'] == query_idx:
                        match_count += 1
                    if rank == 1 and result['index'] == query_idx:
                        correct_count += 1
                    print(f"{rank}. Score: {result['score']:.4f} - Document: {result['document']}")
            else:
                # Use the `score` method for retrievers without `search`
                scores = retriever.score(query)
                for rank, (chunk_idx, score) in enumerate(scores[:3], 1):  # Top 3
                    if chunk_idx == query_idx:
                        match_count += 1
                    if rank == 1 and chunk_idx == query_idx:
                        correct_count += 1
                    print(f"{rank}. Score: {score:.4f} - Document: {chunks[chunk_idx]}")
        correct = correct_count / len(queries)
        print(f"\n CorrectAccuracy: {correct:.2%} ({correct_count}/{len(queries)})")
        match = match_count / len(queries)
        print(f"\n MatchAccuracy: {match:.2%} ({match_count}/{len(queries)})")


if __name__ == "__main__":
    test_retrievers()