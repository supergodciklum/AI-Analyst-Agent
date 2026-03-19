"""
RAG context builder — no ChromaDB dependency.
Uses Azure OpenAI embeddings + cosine similarity for retrieval.
"""

import json
import numpy as np
import pandas as pd
from openai import AzureOpenAI


def _cosine_similarity(a: list, b: list) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


class RAGContext:
    def __init__(
        self,
        api_key: str,
        df: pd.DataFrame,
        dataset_name: str,
        description: str,
        azure_endpoint: str,
        embedding_deployment: str,
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

        self.documents: list = []
        self.embeddings: list = []

        self._index_dataset()

    # ── Indexing ──────────────────────────────────────────────────────────────

    def _index_dataset(self):
        docs = []

        # 1. Overall description
        docs.append(f"Dataset: {self.dataset_name}\n{self.description}")

        # 2. Column-level info
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

        # 3. Sample rows
        sample = self.df.head(5).to_dict(orient="records")
        docs.append(f"Sample rows:\n{json.dumps(sample, default=str, indent=2)}")

        # 4. Column list summary
        docs.append(f"All columns: {', '.join(self.df.columns.tolist())}")

        self.documents = docs
        self.embeddings = self._embed_batch(docs)

    def _embed_batch(self, texts: list) -> list:
        """Embed a list of texts using Azure OpenAI in a single API call."""
        texts = [t[:8000] for t in texts]
        response = self.client.embeddings.create(
            model=self.embedding_deployment,
            input=texts,
        )
        return [item.embedding for item in response.data]

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, n_results: int = 5) -> str:
        """Return top-n most relevant schema chunks for the query."""
        query_emb = self._embed_batch([query])[0]

        scores = [
            (i, _cosine_similarity(query_emb, doc_emb))
            for i, doc_emb in enumerate(self.embeddings)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)

        top_docs = [self.documents[i] for i, _ in scores[:n_results]]
        return "\n\n---\n\n".join(top_docs)

    def full_schema_summary(self) -> str:
        """Compact schema string — used as fallback if embeddings fail."""
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
