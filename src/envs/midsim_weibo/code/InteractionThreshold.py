from __future__ import annotations
import random

class InteractionThreshold:
    def __init__(self, k_same_target: int, k_diff_targets: int, k_keep_following: int):
        self.k_same_target = int(k_same_target)
        self.k_diff_targets = int(k_diff_targets)
        self.k_keep_following = int(k_keep_following)

    @classmethod
    def sample(cls, rng: random.Random) -> "InteractionThreshold":
        return cls(cls.same_targets(rng), cls.diff_targets(rng), cls.keep_following(rng))

    @staticmethod
    def same_targets(rng: random.Random) -> int:
        support = [1, 2]
        probs   = [0.99, 0.01]
        u = rng.random()
        c = 0.0
        for x, p in zip(support, probs):
            c += p
            if u <= c:
                return x
        return support[-1]

    @staticmethod
    def diff_targets(rng: random.Random) -> int:
        support = [1, 2]
        probs   = [0.99, 0.01]
        u = rng.random()
        c = 0.0
        for x, p in zip(support, probs):
            c += p
            if u <= c:
                return x
        return support[-1]

    @staticmethod
    def keep_following(rng: random.Random) -> int:
        support = [1, 2]
        probs   = [0.99, 0.01]
        u = rng.random()
        c = 0.0
        for x, p in zip(support, probs):
            c += p
            if u <= c:
                return x
        return support[-1]


