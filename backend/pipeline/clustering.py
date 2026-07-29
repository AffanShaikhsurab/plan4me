"""Embedding-based deduplication / clustering of knowledge atoms.

Greedy agglomerative clustering by cosine similarity. This is what turns
"lots of atoms" into "38/50 speakers recommended networking": the size of a
cluster (by distinct video) is the support count, and small clusters survive
so minority opinions are preserved rather than averaged away.
"""
from __future__ import annotations

import logging

import numpy as np

from backend.config import get_settings
from backend.llm.embeddings import embed_texts
from backend.schemas import AtomCluster, KnowledgeAtom

logger = logging.getLogger(__name__)


def _cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    unit = vectors / norms
    return unit @ unit.T


def cluster_atoms(atoms: list[KnowledgeAtom]) -> list[AtomCluster]:
    if not atoms:
        return []

    settings = get_settings()
    threshold = settings.dedupe_similarity_threshold

    # Embed the claim text of each atom.
    texts = [a.claim for a in atoms]
    vectors = np.array(embed_texts(texts), dtype=np.float32)
    sim = _cosine_matrix(vectors)

    n = len(atoms)
    assigned = [-1] * n
    next_cluster = 0

    for i in range(n):
        if assigned[i] != -1:
            continue
        assigned[i] = next_cluster
        for j in range(i + 1, n):
            if assigned[j] == -1 and sim[i, j] >= threshold:
                assigned[j] = next_cluster
        next_cluster += 1

    # Build clusters.
    buckets: dict[int, list[KnowledgeAtom]] = {}
    for atom, cid in zip(atoms, assigned):
        atom.cluster_id = cid
        buckets.setdefault(cid, []).append(atom)

    clusters: list[AtomCluster] = []
    for cid, members in buckets.items():
        # representative = highest-confidence atom in the cluster
        rep = max(members, key=lambda a: a.confidence)
        distinct_videos = len({m.video_id for m in members})
        clusters.append(
            AtomCluster(
                cluster_id=cid,
                representative_claim=rep.claim,
                type=rep.type,
                support_count=distinct_videos,
                atoms=members,
            )
        )

    clusters.sort(key=lambda c: c.support_count, reverse=True)
    logger.info("clustered %d atoms into %d clusters", n, len(clusters))
    return clusters
