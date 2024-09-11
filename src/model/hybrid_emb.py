from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import nltk
from openai import OpenAI

# 
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?"
]

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

# 打印密集嵌入
print("Dense Embeddings Shape:", len(dense_embeddings), "x", len(dense_embeddings[0])) 
print("Dense Embeddings (Sample):", dense_embeddings[0])  

print(f"Dense Embeddings Shape: {dense_embeddings.shape}")

# 只显示非零值的部分
print("Sparse Embeddings (Sample):", sparse_embeddings[0])

