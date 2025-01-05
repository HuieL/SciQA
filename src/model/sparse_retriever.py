import numpy as np
import scipy.sparse as sp
import torch
from collections import Counter
import json
import math
from tqdm import tqdm
from typing import List
from transformers import AutoModel, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
from src.model.base_retriever import SparseRetriever


class BM25Retriever(SparseRetriever):
    def __init__(self, k1: float = 1.5, b: float = 0.75, device: str = 'cpu'):
        super().__init__(device)
        self.k1 = k1
        self.b = b
        self._initialized = False

    def _tokenize(self, text: str) -> List[str]:
        # Handle both string and list inputs
        if isinstance(text, list):
            text = ' '.join(text)
        return text.lower().split()

    def index_documents(self, texts: List[str], batch_size: int = 32):
        self.documents = texts
        self.N = len(texts)
        self.doc_freqs = {}
        self.doc_lengths = []
        total_len = 0
        
        # Process all documents
        for text in tqdm(texts, desc="Processing documents"):
            terms = self._tokenize(text)
            self.doc_lengths.append(len(terms))
            total_len += len(terms)
            
            # Update document frequencies for unique terms
            for term in set(terms):
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1
        
        self.avgdl = total_len / self.N if self.N > 0 else 0
        self._initialized = True
        
        # Create sparse matrix representation
        self.doc_embeddings = self.encode(texts)

    def encode(self, texts: List[str], batch_size: int = 32) -> sp.csr_matrix:
        if not self._initialized and len(texts) > 1:
            self.index_documents(texts)
            
        # Build vocabulary
        vocab = sorted(self.doc_freqs.keys())
        term_to_id = {term: idx for idx, term in enumerate(vocab)}
        
        rows, cols, data = [], [], []
        for doc_idx, text in enumerate(texts):
            terms = self._tokenize(text)
            term_freqs = Counter(terms)
            doc_len = len(terms)
            
            for term, freq in term_freqs.items():
                if term in term_to_id:
                    # Calculate BM25 score
                    idf = math.log((self.N - self.doc_freqs.get(term, 0) + 0.5) / 
                                 (self.doc_freqs.get(term, 0) + 0.5) + 1)
                    norm_tf = ((freq * (self.k1 + 1)) / 
                             (freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
                    score = idf * norm_tf
                    
                    rows.append(doc_idx)
                    cols.append(term_to_id[term])
                    data.append(score)
        
        if not vocab: 
            return sp.csr_matrix((len(texts), 0))
                    
        return sp.csr_matrix((data, (rows, cols)), 
                           shape=(len(texts), len(vocab)))
    
class BGEM3Retriever(SparseRetriever):
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(device)
        from FlagEmbedding import BGEM3FlagModel
        self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        self._initialized = False
        
    def index_documents(self, texts: List[str], batch_size: int = 32):
        self.documents = texts
        self.doc_embeddings = self.encode(texts, batch_size)
        self._initialized = True

    def encode(self, texts: List[str], batch_size: int = 32) -> sp.csr_matrix:
        all_weights = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
            batch = texts[i:i + batch_size]
            # Get lexical weights for batch
            batch_outputs = []
            for text in batch:
                if not text.strip():
                    batch_outputs.append(None)
                    continue
                output = self.model.encode(
                    [text],
                    return_dense=False,
                    return_sparse=True
                )
                batch_outputs.append(output['lexical_weights'][0])

            for weights in batch_outputs:
                if weights is None:
                    vocab_size = max(w.shape[0] for w in all_weights) if all_weights else 1
                    all_weights.append(sp.csr_matrix((1, vocab_size)))
                else:
                    all_weights.append(sp.csr_matrix(weights))

        if not all_weights:
            return sp.csr_matrix((len(texts), 1))

        max_width = max(w.shape[1] for w in all_weights)
        padded_weights = []
        for w in all_weights:
            if w.shape[1] < max_width:
                w = sp.hstack([w, sp.csr_matrix((w.shape[0], max_width - w.shape[1]))])
            padded_weights.append(w)

        return sp.vstack(padded_weights)

    def compute_lexical_matching_score(self, query_weights, doc_weights):
        return self.model.compute_lexical_matching_score(query_weights, doc_weights)
    
class Doc2QueryRetriever(SparseRetriever):
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 num_queries: int = 3):
        super().__init__(device)
        self.num_queries = num_queries
        self.tokenizer = T5Tokenizer.from_pretrained("doc2query/msmarco-t5-base-v1")
        self.model = T5ForConditionalGeneration.from_pretrained("doc2query/msmarco-t5-base-v1").to(device)
        self.bm25 = BM25Retriever()

    def _expand_document(self, text: str) -> str:
        inputs = self.tokenizer.encode(text, return_tensors="pt",
                                     max_length=512, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs, max_length=64,
                num_return_sequences=self.num_queries,
                do_sample=True,
                top_k=50, top_p=0.95
            )
        queries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return text + " " + " ".join(queries)

    def index_documents(self, texts: List[str], batch_size: int = 32):
        self.documents = texts
        expanded_texts = [self._expand_document(text) for text in texts]
        self.bm25.index_documents(expanded_texts)
        self.doc_embeddings = self.bm25.doc_embeddings

    def encode(self, texts: List[str], batch_size: int = 32) -> sp.csr_matrix:
        expanded_texts = [self._expand_document(text) for text in texts]
        return self.bm25.encode(expanded_texts)

class SPLADERetriever(SparseRetriever):
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(device)
        self.model = AutoModel.from_pretrained("naver/splade-cocondenser-ensembledistil").to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")

    def encode(self, texts: List[str], batch_size: int = 32) -> sp.csr_matrix:
        rows, cols, data = [], [], []
        for doc_idx, text in enumerate(texts):
            inputs = self.tokenizer(text, padding=True, truncation=True,
                                  max_length=512, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.last_hidden_state.sum(dim=1)
                weights = torch.relu(logits).squeeze(0).cpu().numpy()
                
                nonzero_idx = np.nonzero(weights)[0]
                rows.extend([doc_idx] * len(nonzero_idx))
                cols.extend(nonzero_idx.tolist())
                data.extend(weights[nonzero_idx].tolist())
        
        return sp.csr_matrix((data, (rows, cols)),
                           shape=(len(texts), self.tokenizer.vocab_size))

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
        # "BM25": BM25Retriever(),
        "BGE-M3 Sparse Retriever": BGEM3Retriever(),
        # "Doc2Query": Doc2QueryRetriever(),
        # "SPLADE++": SPLADERetriever()
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
                results = retriever.search(query, chunks, top_k=3)
                for rank, result in enumerate(results, 1):
                    if result['index'] == query_idx:
                        match_count += 1
                    if rank == 1 and result['index'] == query_idx:
                        correct_count += 1
                    # print(f"{rank}. Score: {result['score']:.4f} - Document: {result['document']}")
            else:
                # Use the `score` method for retrievers without `search`
                scores = retriever.score(query)
                for rank, (chunk_idx, score) in enumerate(scores[:3], 1):  # Top 3
                    if chunk_idx == query_idx:
                        match_count += 1
                    if rank == 1 and chunk_idx == query_idx:
                        correct_count += 1
                    # print(f"{rank}. Score: {score:.4f} - Document: {chunks[chunk_idx]}")
        correct = correct_count / len(queries)
        print(f"\n CorrectAccuracy: {correct:.2%} ({correct_count}/{len(queries)})")
        match = match_count / len(queries)
        print(f"\n MatchAccuracy: {match:.2%} ({match_count}/{len(queries)})")


if __name__ == "__main__":
    test_retrievers()