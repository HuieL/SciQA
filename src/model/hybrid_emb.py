from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import nltk
from openai import OpenAI
import numpy as np
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix

# 
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?"
]
def get_hybrid_emb(docs):
    n_components = 1536
    pca_sparse = PCA(n_components=n_components)
    pca_dense = PCA(n_components=n_components)

    combined_embeddings_list = []

    for doc in docs:
        sparse=get_tfidf_embeddings(doc)
        dense=get_openai_embeddings(doc, model="text-embedding-3-small")

        sparse = sparse.reshape(1, -1) if len(sparse.shape) == 1 else sparse
        dense = dense.reshape(1, -1) if len(dense.shape) == 1 else dense

        sparse_reduced = pca_sparse.fit_transform(sparse)
        dense_reduced = pca_dense.fit_transform(dense)
        combined_embeddings = np.hstack((sparse_reduced, dense_reduced))
        combined_embeddings_list.append(combined_embeddings)
    
    combined_embeddings_array = np.vstack(combined_embeddings_list)
    return combined_embeddings_array



# 1.TF-IDF
def get_tfidf_embeddings(docs):
    vectorizer = TfidfVectorizer()
    sparse_embeddings = vectorizer.fit_transform(docs)
    return sparse_embeddings

sparse_embeddings = get_tfidf_embeddings(documents)
print(f"Sparse Embeddings Shape: {sparse_embeddings.shape}")

# 2. bm25
def tokenize_documents(docs):
    return [nltk.word_tokenize(doc.lower()) for doc in docs]

def get_bm25_embeddings(docs):
    tokenized_docs = tokenize_documents(docs)
    bm25 = BM25Okapi(tokenized_docs)
    return bm25

# 3. open_ai dense meb
client = OpenAI()

def get_openai_embeddings(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding


dense_embeddings = get_openai_embeddings(documents)


print("Dense Embeddings Shape:", len(dense_embeddings), "x", len(dense_embeddings[0])) 
print("Dense Embeddings (Sample):", dense_embeddings[0])  

print(f"Dense Embeddings Shape: {dense_embeddings.shape}")

print("Sparse Embeddings (Sample):", sparse_embeddings[0])

