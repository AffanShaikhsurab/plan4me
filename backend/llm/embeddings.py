"""Bedrock Titan text embeddings via boto3.

We use Bedrock embeddings (not sentence-transformers) to avoid a torch
dependency, keeping the core backend lightweight and Py3.14-friendly.
"""
from __future__ import annotations

import json
import logging

import boto3

from backend.config import get_settings

logger = logging.getLogger(__name__)

_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        s = get_settings()
        _CLIENT = boto3.client("bedrock-runtime", region_name=s.aws_region)
    return _CLIENT


def embed_text(text: str) -> list[float]:
    """Return an embedding vector for a single string."""
    s = get_settings()
    body = json.dumps({"inputText": text})
    resp = _client().invoke_model(modelId=s.embedding_model_id, body=body)
    payload = json.loads(resp["body"].read())
    return payload["embedding"]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings (sequential; Titan has no batch endpoint)."""
    return [embed_text(t) for t in texts]
