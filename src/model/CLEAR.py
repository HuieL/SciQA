import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
import numpy as np
from typing import List
from datasets import load_dataset
from rank_bm25 import BM25Okapi
import argparse
import random
import warnings
from tqdm import tqdm


warnings.filterwarnings('ignore')

class MSMarcoDataset(Dataset):
    def __init__(self, queries: List[str], positive_docs: List[str], negative_docs: List[str]):
        self.queries = queries
        self.positive_docs = positive_docs
        self.negative_docs = negative_docs

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.queries[idx], self.positive_docs[idx], self.negative_docs[idx]

def load_msmarco_batch(batch_start, batch_size):
    dataset = load_dataset("ms_marco", "v2.1", split=f"train[{batch_start}:{batch_start + batch_size}]")
    return dataset

def load_msmarco_with_bm25(total_size=500000, batch_size=2000):
    queries = []
    positive_docs = []
    negative_docs = []

    total_batches = total_size // batch_size + (1 if total_size % batch_size > 0 else 0)
    for batch_index in tqdm(range(total_batches), desc="Processing Batches", unit="batch"):
        batch_start = batch_index * batch_size
        batch = load_msmarco_batch(batch_start, batch_size)

        # Build BM25 index for the current batch
        batch_corpus = [passage for example in batch for passage in example["passages"]["passage_text"]]
        tokenized_corpus = [doc.split() for doc in batch_corpus]
        bm25 = BM25Okapi(tokenized_corpus)

        # Collect queries and compute BM25 scores
        for example in batch:
            query = example["query"]
            is_selected = example["passages"]["is_selected"]
            passage_texts = example["passages"]["passage_text"]

            # Extract positive passages
            positives = [text for selected, text in zip(is_selected, passage_texts) if selected]

            if positives:
                query_tokens = query.split()
                scores = bm25.get_scores(query_tokens)
                ranked_passages = [
                    batch_corpus[i] for i in np.argsort(scores)[::-1][:1000] if batch_corpus[i] not in positives
                ]

                if ranked_passages:
                    random_negative = random.choice(ranked_passages)
                    queries.append(query)
                    positive_docs.append(positives[0]) 
                    negative_docs.append(random_negative)  
    return queries, positive_docs, negative_docs

class ResidualDenseRetriever(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", device="cuda"):
        super().__init__()
        self.encoder = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.device = device
        self.to(device)  # Move model to the correct device

    def forward(self, texts: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt", max_length=512
        ).to(self.device)

        outputs = self.encoder(**tokens)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling
        return embeddings.to(self.device)

def residual_loss(lexical_scores, emb_scores, lambda_train=0.1, margin=1.0):
    """
    Compute the residual loss for the dense retriever.
    """
    residual_margin = margin - lambda_train * (lexical_scores[:, 0] - lexical_scores[:, 1])
    loss = torch.relu(residual_margin - (emb_scores[:, 0] - emb_scores[:, 1])).mean()
    return loss

def train_dense_retriever(
    retriever,
    dataloader,
    optimizer,
    epochs=3,
    lambda_train=0.1,
    device="cuda",
    save_path="clear_model.pt"
):
    retriever.to(device)
    retriever.train()

    for epoch in range(epochs):
        total_loss = 0
        for queries, positive_docs, negative_docs in dataloader:
            queries = [q for q in queries]
            positive_docs = [p for p in positive_docs]
            negative_docs = [n for n in negative_docs]

            query_embeddings = retriever(queries)
            pos_embeddings = retriever(positive_docs)
            neg_embeddings = retriever(negative_docs)

            # Fake lexical scores (random for demonstration)
            lexical_scores = torch.rand((len(queries), 2), device=device)

            # Compute embedding scores
            emb_scores = torch.cat(
                [
                    (query_embeddings * pos_embeddings).sum(dim=1, keepdim=True),
                    (query_embeddings * neg_embeddings).sum(dim=1, keepdim=True),
                ],
                dim=1,
            )

            loss = residual_loss(lexical_scores, emb_scores, lambda_train=lambda_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss:.4f}")

    torch.save(retriever.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CLEAR model.")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs (from paper)")
    parser.add_argument("--batch_size", type=int, default=28, help="Batch size for training (from paper)")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate (from paper)")
    parser.add_argument("--lambda_train", type=float, default=0.1, help="Lambda for residual learning (from paper)")
    parser.add_argument("--save_path", type=str, default="clear_model.pt", help="Path to save the trained model")
    args = parser.parse_args()

    print("Loading MS MARCO dataset with BM25 negatives...")
    queries, positive_docs, negative_docs = load_msmarco_with_bm25()

    dataset = MSMarcoDataset(queries, positive_docs, negative_docs)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    retriever = ResidualDenseRetriever()
    optimizer = torch.optim.AdamW(retriever.parameters(), lr=args.learning_rate)

    train_dense_retriever(
        retriever,
        dataloader,
        optimizer,
        epochs=args.epochs,
        lambda_train=args.lambda_train,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_path=args.save_path,
    )
