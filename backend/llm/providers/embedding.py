"""Concrete embedding providers.

Each returns unit-comparable vectors for cosine similarity, and each declares
the similarity cutoff that suits its own vector space - a lexical space and a
semantic space are not interchangeable at a fixed threshold.
"""
from __future__ import annotations

import json
import math
import re
import zlib
from collections import Counter
from typing import Any

import numpy as np

from backend.llm.providers.base import EmbeddingProvider, ProviderError
from backend.llm.providers.registry import register_embedding


@register_embedding
class BedrockEmbeddingProvider(EmbeddingProvider):
    """Amazon Titan text embeddings through boto3."""

    name = "bedrock"
    default_similarity_threshold = 0.82

    _client: Any = None

    def _runtime(self) -> Any:
        if self._client is None:
            import boto3

            self._client = boto3.client(
                "bedrock-runtime", region_name=self._settings.aws_region
            )
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Titan has no batch endpoint, so this is sequential by necessity.
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        resp = self._runtime().invoke_model(
            modelId=self._settings.embedding_model_id,
            body=json.dumps({"inputText": text}),
        )
        return json.loads(resp["body"].read())["embedding"]

    def describe(self) -> dict:
        return {**super().describe(), "model": self._settings.embedding_model_id}


@register_embedding
class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Any OpenAI-compatible /embeddings endpoint."""

    name = "openai"
    default_similarity_threshold = 0.82

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ProviderError("openai is not installed. Run: pip install openai") from exc

        key = self._settings.openai_embedding_api_key.strip()
        if not key:
            raise ProviderError(
                "EMBEDDING_PROVIDER=openai requires OPENAI_EMBEDDING_API_KEY."
            )
        client = OpenAI(api_key=key, base_url=self._settings.openai_embedding_base_url)
        resp = client.embeddings.create(
            model=self._settings.openai_embedding_model, input=texts
        )
        # Preserve request order regardless of how the server returns them.
        return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]

    def describe(self) -> dict:
        return {**super().describe(), "model": self._settings.openai_embedding_model}


@register_embedding
class LocalEmbeddingProvider(EmbeddingProvider):
    """Hashed TF-IDF. No key, no network, no torch.

    Good enough to collapse reworded duplicates, and weaker than a semantic
    model at matching paraphrases that share no vocabulary. The threshold was
    calibrated on paraphrase pairs: 0.35 merged 4/5 true duplicates with zero
    false merges across 185 pairs, where the non-duplicate ceiling was 0.25.
    """

    name = "local"
    default_similarity_threshold = 0.35

    _DIM = 2048
    _TOKEN_RE = re.compile(r"[a-z0-9]+")

    def similarity_threshold(self) -> float:
        """Tunable via LOCAL_DEDUPE_SIMILARITY_THRESHOLD."""
        return self._settings.local_dedupe_similarity_threshold

    def embed(self, texts: list[str]) -> list[list[float]]:
        docs = [self._features(t) for t in texts]
        n = len(docs)

        # Document frequency over this batch. cluster_atoms() embeds the whole
        # corpus in one call, so IDF is corpus-wide in practice.
        df: Counter[str] = Counter()
        for feats in docs:
            df.update(set(feats))

        out = np.zeros((n, self._DIM), dtype=np.float32)
        for i, feats in enumerate(docs):
            for feat, tf in Counter(feats).items():
                idf = math.log((1.0 + n) / (1.0 + df[feat])) + 1.0
                out[i, self._bucket(feat)] += tf * idf

        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        return (out / norms).tolist()

    @classmethod
    def _features(cls, text: str) -> list[str]:
        """Word unigrams plus bigrams, so word order carries some weight."""
        toks = cls._TOKEN_RE.findall(text.lower())
        return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]

    @classmethod
    def _bucket(cls, feature: str) -> int:
        # crc32 keeps bucketing stable across processes, unlike hash().
        return zlib.crc32(feature.encode("utf-8")) % cls._DIM
