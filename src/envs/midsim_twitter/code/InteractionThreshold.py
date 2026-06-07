from __future__ import annotations
import random

class InteractionThreshold:
    def __init__(self, k_same_target: int, k_diff_targets: int, k_keep_following: int, propagation_type: str, mention_type: str):
        self.k_same_target = int(k_same_target)
        self.k_diff_targets = int(k_diff_targets)
        self.k_keep_following = int(k_keep_following)
        self.propagation_type = propagation_type
        self.mention_type = mention_type

    @classmethod
    def sample(cls, rng: random.Random) -> "InteractionThreshold":
        return cls(cls.same_targets(rng), cls.diff_targets(rng), cls.keep_following(rng), cls.propagation_type(rng), cls.mention_type(rng))

    @staticmethod
    def same_targets(rng: random.Random) -> int:
        support = [1, 2, 3]
        probs = [0.6, 0.3, 0.1]
        u = rng.random()
        c = 0.0
        for x, p in zip(support, probs):
            c += p
            if u <= c:
                return x
        return support[-1]

    @staticmethod
    def diff_targets(rng: random.Random) -> int:
        support = [1, 2, 3, 4, 5]
        probs   = [0.4, 0.3, 0.2, 0.05, 0.05]
        u = rng.random()
        c = 0.0
        for x, p in zip(support, probs):
            c += p
            if u <= c:
                return x
        return support[-1]

    @staticmethod
    def keep_following(rng: random.Random) -> int:
        support = [0, 1]
        probs   = [0.9961, 0.0039]
        u = rng.random()
        c = 0.0
        for x, p in zip(support, probs):
            c += p
            if u <= c:
                return x
        return support[-1]

    def propagation_type(rng: random.Random) -> str:
        support = ["retweet", "reply", "quote"]
        probs   = [0.7, 0.03, 0.27]
        u = rng.random()
        c = 0.0
        for x, p in zip(support, probs):
            c += p
            if u <= c:
                return x

    def mention_type(rng: random.Random) -> str:
        support = ["retweet", "reply", "quote"]
        probs   = [0.0, 0.9, 0.1]
        u = rng.random()
        c = 0.0
        for x, p in zip(support, probs):
            c += p
            if u <= c:
                return x