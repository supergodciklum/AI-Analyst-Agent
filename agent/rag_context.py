"""
RAG context builder.
Uses Azure OpenAI embeddings + cosine similarity.
Caches embeddings to disk so re-loading the same dataset skips the API call.

Features:
- Optional cost_tracker to record embedding API usage.
"""

import json
import os
import hashlib
import pickle
import numpy as np
import pandas as pd
from openai import AzureOpenAI


def _cosine_similarity(a: list, b: list) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def _cache_key(dataset_name: str, df: pd.DataFrame) -> str:
    content = f"{dataset_name}|{df.shape}|{'|'.join(df.columns.tolist())}"
    return hashlib.md5(content.encode()).hexdigest()


def _cache_path(key: str) -> str:
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"emb_{key}.pkl")


class RAGContext:
    def __init__(
        self,
        api_key: str,
        df: pd.DataFrame,
        dataset_name: str,
        description: str,
        azure_endpoint: str,
        embedding_deployment: str,
        cost_tracker=None,
    ):
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version="2024-02-01",
        )
        self.df = df
        self.dataset_name = dataset_name
        self.description = description
        self.embedding_deployment = embedding_deployment
        self.cost_tracker = cost_tracker

        self.documents: list = []
        self.embeddings: list = []
        self._from_cache = False

        self._index_dataset()

    # ── Indexing ──────────────────────────────────────────────────────────────

    def _index_dataset(self):
        key  = _cache_key(self.dataset_name, self.df)
        path = _cache_path(key)

        # Try loading from disk cache first
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    cached = pickle.load(f)
                self.documents = cached["documents"]
                self.embeddings = cached["embeddings"]
                self._from_cache = True
                return
            except Exception:
                pass  # cache corrupted — rebuild below

        # Build documents
        docs = []
        docs.append(f"Dataset: {self.dataset_name}\n{self.description}")

        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            if self.df[col].dtype in ["object", "category"]:
                uniq = self.df[col].dropna().unique().tolist()[:10]
                docs.append(f"Column '{col}': categorical, dtype={dtype}, unique values={uniq}")
            else:
                s = self.df[col].describe()
                docs.append(
                    f"Column '{col}': numeric, dtype={dtype}, "
                    f"min={s['min']:.2f}, max={s['max']:.2f}, "
                    f"mean={s['mean']:.2f}, std={s['std']:.2f}"
                )

        sample = self.df.head(5).to_dict(orient="records")
        docs.append(f"Sample rows:\n{json.dumps(sample, default=str, indent=2)}")
        docs.append(f"All columns: {', '.join(self.df.columns.tolist())}")

        self.documents = docs
        self.embeddings = self._embed_batch(docs)

        # Persist to cache
        try:
            with open(path, "wb") as f:
                pickle.dump({"documents": docs, "embeddings": self.embeddings}, f)
        except Exception:
            pass

    def _embed_batch(self, texts: list) -> list:
        texts = [t[:8000] for t in texts]
        response = self.client.embeddings.create(
            model=self.embedding_deployment,
            input=texts,
        )
        # Record embedding cost if tracker available
        if self.cost_tracker is not None:
            try:
                total_input = response.usage.total_tokens
                self.cost_tracker.record(
                    model=self.embedding_deployment,
                    input_tokens=total_input,
                    output_tokens=0,
                    operation="embed",
                )
            except Exception:
                pass
        return [item.embedding for item in response.data]

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, n_results: int = 5) -> str:
        query_emb = self._embed_batch([query])[0]
        scores = [
            (i, _cosine_similarity(query_emb, doc_emb))
            for i, doc_emb in enumerate(self.embeddings)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        top_docs = [self.documents[i] for i, _ in scores[:n_results]]
        return "\n\n---\n\n".join(top_docs)

    def full_schema_summary(self) -> str:
        lines = [f"Dataset: {self.dataset_name}", f"Shape: {self.df.shape}", "Columns:"]
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            if self.df[col].dtype in ["object", "category"]:
                uniq = self.df[col].dropna().unique().tolist()[:5]
                lines.append(f"  - {col} ({dtype}): {uniq}")
            else:
                lines.append(
                    f"  - {col} ({dtype}): "
                    f"min={self.df[col].min():.2f} max={self.df[col].max():.2f}"
                )
        return "\n".join(lines)
