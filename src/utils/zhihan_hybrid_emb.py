from sklearn.feature_extraction.text import TfidfVectorizer
# from rank_bm25 import BM25Okapi
import nltk
from openai import OpenAI
import numpy as np
from sklearn.decomposition import PCA

#using PCA to combine embs, 
def get_hybrid_emb(docs):
    n_components = min(1536, len(docs)-1)
    
    pca_sparse = PCA(n_components=n_components)
    pca_dense = PCA(n_components=n_components)
    
    sparse_embeddings = get_tfidf_embeddings(docs) 
    dense_embeddings = np.array([get_openai_embeddings(doc) for doc in docs]) 

    sparse_reduced = pca_sparse.fit_transform(sparse_embeddings)
    dense_reduced = pca_dense.fit_transform(dense_embeddings)

    combined_embeddings = np.hstack((sparse_reduced, dense_reduced))
    
    return combined_embeddings



# 1.TF-IDF
def get_tfidf_embeddings(docs):
    vectorizer = TfidfVectorizer()
    sparse_embeddings = vectorizer.fit_transform(docs)
    return sparse_embeddings


# 2. bm25
# def tokenize_documents(docs):
#     return [nltk.word_tokenize(doc.lower()) for doc in docs]

# def get_bm25_embeddings(docs):
#     tokenized_docs = tokenize_documents(docs)
#     bm25 = BM25Okapi(tokenized_docs)
#     return bm25

# 3. open_ai dense meb
client = OpenAI()

def get_openai_embeddings(texts, model="text-embedding-3-small"):
#    text = text.replace("\n", " ")
   return client.embeddings.create(input = texts, model=model).data[0].embedding