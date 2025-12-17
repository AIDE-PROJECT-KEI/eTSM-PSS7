# eTSM-PSS7
eTSM-PSS7 is a research prototype for evaluating self-maintenance in large language models. It implements an external persona and memory exoskeleton with multi-timescale updates, enabling quantitative measurement of long-horizon consistency, contradiction resilience, and persona stability without modifying model weights.
# src/etsm/core.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math
import time


@dataclass
class EpisodeTurn:
    t: int
    user: str
    raw: str
    final: str
    scores: List[float]  # PSS-n scores (e.g., 7 dims)
    v: List[float]
    theta: List[float]
    memory_add: bool
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryItem:
    text: str
    embedding: Optional[List[float]] = None
    importance: float = 0.0
    ts: float = field(default_factory=lambda: time.time())


class ExternalMemory:
    """
    Minimal bounded external memory with importance-based admission + eviction.
    """
    def __init__(self, capacity: int = 128):
        self.capacity = capacity
        self.items: List[MemoryItem] = []

    def add(self, item: MemoryItem) -> None:
        self.items.append(item)
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        if len(self.items) <= self.capacity:
            return
        # evict lowest-importance first
        self.items.sort(key=lambda x: x.importance, reverse=True)
        self.items = self.items[: self.capacity]

    def retrieve(self, query: str, k: int = 5) -> List[MemoryItem]:
        # Placeholder retrieval: simple recency/importance mix.
        # Replace with embedding search later.
        if not self.items:
            return []
        ranked = sorted(
            self.items,
            key=lambda x: (x.importance, x.ts),
            reverse=True
        )
        return ranked[:k]


def l2_norm(vec: List[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class ExoskeletonConfig:
    pss_dim: int = 7
    alpha: float = 0.08          # persona update rate
    beta: float = 0.002          # ultra-slow update rate (beta << alpha)
    theta_dim: int = 7
    theta_period: int = 50       # update theta every T turns
    memory_capacity: int = 128
    memory_k: int = 5
    memory_importance_threshold: float = 0.55


class ETSMExoskeleton:
    """
    Minimal eTSM-PSS7 exoskeleton:
    - persona vector v_t (slow)
    - external memory M_t (bounded)
    - ultra-slow parameter theta_t (ultra-slow)
    This module does not modify the base model weights.
    """
    def __init__(self, cfg: ExoskeletonConfig):
        self.cfg = cfg
        self.v: List[float] = [0.0] * cfg.pss_dim
        self.theta: List[float] = [0.0] * cfg.theta_dim
        self.M = ExternalMemory(capacity=cfg.memory_capacity)
        self.turn: int = 0

    # ----- scoring / evaluation hooks -----

    def score_pss(self, user: str, raw: str, final: str) -> List[float]:
        """
        Returns PSS-n axis scores in [-1, +1].
        Placeholder: deterministic heuristic stub.
        Replace with LLM-as-judge or NLI+rules later.
        """
        # Example heuristics (very minimal):
        # axis0: consistency preference proxy (penalize explicit contradictions markers)
        neg = sum(tok in final.lower() for tok in ["i disagree with myself", "actually i was wrong", "contradict"])
        axis0 = 1.0 - 0.5 * neg

        # axis1: cooperation proxy (politeness markers)
        pos = sum(tok in final.lower() for tok in ["sure", "happy to", "i can", "let's"])
        axis1 = clamp(-0.2 + 0.2 * pos, -1.0, 1.0)

        # fill remaining axes with zeros for now
        scores = [0.0] * self.cfg.pss_dim
        scores[0] = clamp(axis0, -1.0, 1.0)
        if self.cfg.pss_dim > 1:
            scores[1] = axis1
        return scores

    def memory_importance(self, user: str, final: str, scores: List[float]) -> float:
        """
        Importance in [0, 1] deciding whether to store an episode.
        Placeholder: combine axis0 (consistency) and presence of commitments.
        """
        commitment = any(kw in final.lower() for kw in ["i will", "i won't", "my policy", "i refuse", "i commit"])
        base = 0.5 + 0.25 * (scores[0] if scores else 0.0)
        if commitment:
            base += 0.2
        return clamp(base, 0.0, 1.0)

    # ----- update rules -----

    def update_persona(self, scores: List[float]) -> None:
        """
        v_t[i] = (1 - alpha) v_{t-1}[i] + alpha * score_i
        """
        a = self.cfg.alpha
        for i in range(min(len(self.v), len(scores))):
            self.v[i] = (1.0 - a) * self.v[i] + a * scores[i]

    def update_theta(self) -> None:
        """
        theta_t = (1 - beta) theta_{t-1} + beta * f(v_t)
        Here f(v_t) is identity for minimal prototype.
        """
        b = self.cfg.beta
        for i in range(min(len(self.theta), len(self.v))):
            self.theta[i] = (1.0 - b) * self.theta[i] + b * self.v[i]

    # ----- integration hooks -----

    def build_context(self, user: str) -> str:
        """
        Builds exoskeleton context injected into the base model prompt.
        Keep this compact for low-latency usage.
        """
        mem = self.M.retrieve(user, k=self.cfg.memory_k)
        mem_block = "\n".join([f"- {m.text}" for m in mem]) if mem else "(none)"
        v_str = ", ".join([f"{x:+.3f}" for x in self.v])
        th_str = ", ".join([f"{x:+.3f}" for x in self.theta])

        return (
            "EXOSKELETON_STATE\n"
            f"persona_vector_v: [{v_str}]\n"
            f"ultra_slow_theta: [{th_str}]\n"
            "relevant_memory:\n"
            f"{mem_block}\n"
        )

    def post_process(self, raw: str, user: str) -> str:
        """
        Output correction stage placeholder.
        For minimal prototype: return raw as final.
        Later: re-rank candidates using consistency/memory alignment.
        """
        return raw

    def step(
        self,
        user: str,
        raw_response: str,
        final_response: Optional[str] = None
    ) -> Tuple[str, EpisodeTurn]:
        """
        One turn update. final_response can be provided if a later stage rewrites raw.
        """
        t = self.turn
        final = final_response if final_response is not None else self.post_process(raw_response, user)

        scores = self.score_pss(user, raw_response, final)
        self.update_persona(scores)

        # memory decision
        imp = self.memory_importance(user, final, scores)
        add = imp >= self.cfg.memory_importance_threshold
        if add:
            self.M.add(MemoryItem(text=f"U: {user}\nA: {final}", importance=imp))

        # ultra-slow update on schedule
        if (t + 1) % self.cfg.theta_period == 0:
            self.update_theta()

        turn_log = EpisodeTurn(
            t=t,
            user=user,
            raw=raw_response,
            final=final,
            scores=scores,
            v=list(self.v),
            theta=list(self.theta),
            memory_add=add,
            meta={"importance": imp}
        )
        self.turn += 1
        return final, turn_log
