from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .osnet import OSNetAdapter


@dataclass
class Identity:
    identity_id: int
    embedding: np.ndarray
    last_track_id: int


class IdentityMatcher:
    def __init__(self, model: OSNetAdapter, threshold: float = 0.55):
        self.model = model
        self.threshold = threshold
        self.identities: dict[int, Identity] = {}
        self.next_identity = 1

    def match(self, embedding: np.ndarray, track_id: int) -> tuple[int, float, bool]:
        best_id, best_score = None, -1.0
        for iid, identity in self.identities.items():
            score = self.model.similarity(embedding, identity.embedding)
            if score > best_score:
                best_id, best_score = iid, score
        if best_id is not None and best_score >= self.threshold:
            self.identities[best_id].embedding = embedding
            self.identities[best_id].last_track_id = track_id
            return best_id, best_score, True
        iid = self.next_identity
        self.next_identity += 1
        self.identities[iid] = Identity(iid, embedding, track_id)
        return iid, 1.0, False
