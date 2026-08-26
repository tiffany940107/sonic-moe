from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class CustomerConfig:
    global_tokens: int = 16_384
    top_k: int = 24
    hidden_size: int = 2_560
    intermediate_size: int = 1_024
    global_experts: int = 768
    ep_size: int = 4
    seed: int = 42

    @property
    def tokens_per_rank(self) -> int:
        if self.global_tokens % self.ep_size:
            raise ValueError("global_tokens must divide ep_size")
        return self.global_tokens // self.ep_size

    @property
    def local_experts(self) -> int:
        if self.global_experts % self.ep_size:
            raise ValueError("global_experts must divide ep_size")
        return self.global_experts // self.ep_size

    @property
    def global_pairs(self) -> int:
        return self.global_tokens * self.top_k

    @property
    def ideal_pairs_per_rank(self) -> int:
        return self.global_pairs // self.ep_size

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out.update(
            tokens_per_rank=self.tokens_per_rank,
            local_experts=self.local_experts,
            global_pairs=self.global_pairs,
            ideal_pairs_per_rank=self.ideal_pairs_per_rank,
        )
        return out


def percentile(values: Iterable[float], q: float) -> float:
    vals = sorted(float(v) for v in values)
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def summarize_ms(values: Iterable[float]) -> dict[str, float]:
    vals = [float(v) for v in values]
    return {
        "count": len(vals),
        "mean_ms": sum(vals) / len(vals) if vals else float("nan"),
        "p50_ms": percentile(vals, 0.50),
        "p95_ms": percentile(vals, 0.95),
        "p99_ms": percentile(vals, 0.99),
        "min_ms": min(vals, default=float("nan")),
        "max_ms": max(vals, default=float("nan")),
    }


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
