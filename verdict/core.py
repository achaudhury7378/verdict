"""
verdict/core.py — The engine. Experiment + Analysis.
Parallel execution for speed.
"""

import time
import random
import itertools
import numpy as np
from scipy import stats
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Optional, Callable


# ═══════════════════════════════════════════════
#  PRICING — OpenAI models (cost per token USD)
# ═══════════════════════════════════════════════

PRICING = {
    "gpt-4o": {
        "input": 2.50 / 1_000_000,
        "cached_input": 1.25 / 1_000_000,
        "output": 10.00 / 1_000_000,
    },
    "gpt-4o-mini": {
        "input": 0.15 / 1_000_000,
        "cached_input": 0.075 / 1_000_000,
        "output": 0.60 / 1_000_000,
    },
    "gpt-4.1": {
        "input": 2.00 / 1_000_000,
        "cached_input": 0.50 / 1_000_000,
        "output": 8.00 / 1_000_000,
    },
    "gpt-4.1-mini": {
        "input": 0.40 / 1_000_000,
        "cached_input": 0.10 / 1_000_000,
        "output": 1.60 / 1_000_000,
    },
    "gpt-4.1-nano": {
        "input": 0.10 / 1_000_000,
        "cached_input": 0.025 / 1_000_000,
        "output": 0.40 / 1_000_000,
    },
    "gpt-5.4": {
        "input": 2.50 / 1_000_000,
        "cached_input": 0.25 / 1_000_000,
        "output": 15.00 / 1_000_000,
    },
    "gpt-5.4-mini": {
        "input": 0.75 / 1_000_000,
        "cached_input": 0.075 / 1_000_000,
        "output": 4.50 / 1_000_000,
    },
    "gpt-5.4-nano": {
        "input": 0.20 / 1_000_000,
        "cached_input": 0.020 / 1_000_000,
        "output": 1.25 / 1_000_000,
    },
    "gpt-5.2": {
        "input": 1.75 / 1_000_000,
        "cached_input": 0.175 / 1_000_000,
        "output": 14.00 / 1_000_000,
    },
    "gpt-5.1": {
        "input": 1.25 / 1_000_000,
        "cached_input": 0.125 / 1_000_000,
        "output": 10.00 / 1_000_000,
    },
    "gpt-5": {
        "input": 1.25 / 1_000_000,
        "cached_input": 0.125 / 1_000_000,
        "output": 10.00 / 1_000_000,
    },
    "gpt-5-mini": {
        "input": 0.25 / 1_000_000,
        "cached_input": 0.025 / 1_000_000,
        "output": 2.00 / 1_000_000,
    },
    "gpt-5-nano": {
        "input": 0.05 / 1_000_000,
        "cached_input": 0.005 / 1_000_000,
        "output": 0.40 / 1_000_000,
    },
    "o3": {
        "input": 2.00 / 1_000_000,
        "cached_input": 0.50 / 1_000_000,
        "output": 8.00 / 1_000_000,
    },
    "o3-mini": {
        "input": 1.10 / 1_000_000,
        "cached_input": 0.55 / 1_000_000,
        "output": 4.40 / 1_000_000,
    },
    "o4-mini": {
        "input": 1.10 / 1_000_000,
        "cached_input": 0.275 / 1_000_000,
        "output": 4.40 / 1_000_000,
    },
}


# ═══════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════


@dataclass
class Variant:
    """
    One model to test.

    Parameters
    ----------
    name : str
        Display name. e.g., "gpt-4o"
    fn : Callable
        (config, query) → str or dict
        str: verdict estimates tokens
        dict: {"output": "...", "usage": {"input_tokens": N, "output_tokens": N}}
    config : dict
        Passed as first arg to fn.
    model : str, optional
        For pricing lookup.
    pricing : dict, optional
        Override: {"input": per_token, "output": per_token}
    """

    name: str
    fn: Callable
    config: dict = field(default_factory=dict)
    model: Optional[str] = None
    pricing: Optional[dict] = None

    def get_pricing(self) -> dict:
        if self.pricing:
            pricing = dict(self.pricing)
        elif self.model and self.model in PRICING:
            pricing = dict(PRICING[self.model])
        elif self.name in PRICING:
            pricing = dict(PRICING[self.name])
        else:
            pricing = {"input": 0.0, "cached_input": 0.0, "output": 0.0}

        pricing.setdefault("cached_input", pricing.get("input", 0.0))
        pricing.setdefault("input", 0.0)
        pricing.setdefault("output", 0.0)
        return pricing


@dataclass
class RunResult:
    """One execution of one variant on one query."""

    variant: str
    query: Any
    query_idx: int
    output: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    cost: float
    error: Optional[str]
    score: float = 0.0


# ═══════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════


def _get_query_text(query) -> str:
    """Extract display text from any query type."""
    if isinstance(query, str):
        return query
    if isinstance(query, dict):
        return query.get("query", str(query))
    return str(query)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate. ~1.35 tokens per word."""
    if not text:
        return 0
    return max(1, int(len(text.split()) * 1.35))


def _usage_get(usage: Any, *path, default=0):
    """Read usage values from either dicts or SDK usage objects."""
    current = usage
    for key in path:
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
    return default if current is None else current


# ═══════════════════════════════════════════════
#  EXPERIMENT
# ═══════════════════════════════════════════════


class Experiment:
    """
    Compare N models on M queries with statistical rigor.

    Parameters
    ----------
    name : str
        Experiment name for display.
    variants : list[Variant]
        Models to compare. At least 2.
    queries : list
        Test inputs. Strings or dicts.
    judge : Judge
        Evaluator that scores outputs.
    runs : int
        Runs per variant per query. Default 20.
    monthly_volume : int
        For cost projection. Default 10,000.
    seed : int, optional
        For reproducible run ordering.
    max_workers : int, optional
        Max parallel threads. Default 10.
        Higher = faster but may hit API rate limits.
    """

    def __init__(
        self,
        name: str = "verdict experiment",
        variants: list = None,
        queries: list = None,
        judge=None,
        runs: int = 20,
        monthly_volume: int = 10_000,
        seed: Optional[int] = None,
        max_workers: int = 10,
    ):
        self.name = name
        self.variants = variants or []
        self.queries = queries or []
        self.judge = judge
        self.runs = runs
        self.monthly_volume = monthly_volume
        self.seed = seed
        self.max_workers = max_workers

        if len(self.variants) < 2:
            raise ValueError("Need at least 2 variants to compare.")
        if not self.queries:
            raise ValueError("Need at least 1 query.")
        if self.judge is None:
            raise ValueError("Need a judge. Use Judge(criteria=[...]).")

    def run(self) -> "Analysis":
        """Execute experiment in parallel and return Analysis."""

        # Build queue
        queue = []
        for variant in self.variants:
            for q_idx, query in enumerate(self.queries):
                for _ in range(self.runs):
                    queue.append((variant, query, q_idx))

        if self.seed is not None:
            random.seed(self.seed)
        random.shuffle(queue)

        total = len(queue)
        print(f"\n  verdict — {self.name}")
        print(
            f"  {len(self.variants)} variants × {len(self.queries)} queries "
            f"× {self.runs} runs = {total} total\n"
        )

        # ── Execute all runs in parallel ─────────
        results = [None] * total
        done_count = 0

        print("  Running experiments...")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {}
            for i, (variant, query, q_idx) in enumerate(queue):
                future = executor.submit(self._execute, variant, query, q_idx)
                future_to_idx[future] = i

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
                done_count += 1
                self._progress(done_count, total)

        self._progress(total, total, done=True)

        # ── Judge all outputs in parallel ────────
        done_count = 0

        print("  Judging outputs...")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {}
            for i, result in enumerate(results):
                if result.error is None:
                    query_text = _get_query_text(result.query)
                    future = executor.submit(self.judge.score, result.output, query_text)
                    future_to_idx[future] = i
                else:
                    result.score = 0.0
                    done_count += 1

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx].score = future.result()
                done_count += 1
                self._progress(done_count, total)

        self._progress(total, total, done=True)

        return Analysis(
            name=self.name,
            results=results,
            variants=self.variants,
            queries=self.queries,
            monthly_volume=self.monthly_volume,
        )

    def _execute(self, variant, query, q_idx):
        """Execute ONE run."""
        pricing = variant.get_pricing()
        start = time.perf_counter()

        try:
            raw = variant.fn(variant.config, query)

            if isinstance(raw, dict):
                output = raw.get("output", str(raw))
                usage = raw.get("usage", {})
                in_tok = _usage_get(usage, "input_tokens")
                if not in_tok:
                    in_tok = _usage_get(usage, "prompt_tokens")

                out_tok = _usage_get(usage, "output_tokens")
                if not out_tok:
                    out_tok = _usage_get(usage, "completion_tokens")

                cached_in_tok = _usage_get(usage, "cached_input_tokens")
                if not cached_in_tok:
                    cached_in_tok = _usage_get(
                        usage, "prompt_tokens_details", "cached_tokens"
                    )
                if not cached_in_tok:
                    cached_in_tok = _usage_get(
                        usage, "input_tokens_details", "cached_tokens"
                    )
            else:
                output = str(raw)
                in_tok = _estimate_tokens(_get_query_text(query))
                out_tok = _estimate_tokens(output)
                cached_in_tok = 0

            error = None

        except Exception as e:
            output = ""
            in_tok = 0
            out_tok = 0
            cached_in_tok = 0
            error = str(e)

        latency = (time.perf_counter() - start) * 1000
        cached_in_tok = max(0, min(int(cached_in_tok), int(in_tok)))
        uncached_in_tok = max(0, int(in_tok) - cached_in_tok)
        cost = (
            uncached_in_tok * pricing["input"]
            + cached_in_tok * pricing["cached_input"]
            + int(out_tok) * pricing["output"]
        )

        return RunResult(
            variant=variant.name,
            query=query,
            query_idx=q_idx,
            output=output,
            latency_ms=latency,
            input_tokens=int(in_tok),
            output_tokens=int(out_tok),
            cached_input_tokens=cached_in_tok,
            cost=cost,
            error=error,
        )

    def _progress(self, current, total, done=False):
        pct = current / total
        bar = "█" * int(pct * 40) + "░" * (40 - int(pct * 40))
        end = " ✓\n" if done else ""
        print(f"\r  [{bar}] {current}/{total}{end}", end="", flush=True)


# ═══════════════════════════════════════════════
#  ANALYSIS
# ═══════════════════════════════════════════════


class Analysis:
    """Statistics and comparisons. Created by Experiment.run()."""

    def __init__(self, name, results, variants, queries, monthly_volume):
        self.name = name
        self.results = results
        self.variants = variants
        self.queries = queries
        self.monthly_volume = monthly_volume

        # Group by variant
        self.by_variant = {}
        for v in variants:
            self.by_variant[v.name] = [r for r in results if r.variant == v.name]

        # Pre-compute arrays (successful runs only)
        self.scores = {}
        self.latencies = {}
        self.costs = {}
        for vname, runs in self.by_variant.items():
            ok = [r for r in runs if r.error is None]
            self.scores[vname] = (
                np.array([r.score for r in ok]) if ok else np.array([0.0])
            )
            self.latencies[vname] = (
                np.array([r.latency_ms for r in ok]) if ok else np.array([0.0])
            )
            self.costs[vname] = (
                np.array([r.cost for r in ok]) if ok else np.array([0.0])
            )

    def summary(self, vname: str) -> dict:
        """Full metrics for one variant."""
        s = self.scores[vname]
        l = self.latencies[vname]
        c = self.costs[vname]
        runs = self.by_variant[vname]
        errs = sum(1 for r in runs if r.error is not None)
        n = len(s)
        ci = 1.96 * np.std(s, ddof=1) / np.sqrt(n) if n > 1 else 0
        ok_runs = [r for r in runs if r.error is None]
        total_input = sum(r.input_tokens for r in ok_runs)
        total_cached = sum(r.cached_input_tokens for r in ok_runs)
        cache_hit_rate = total_cached / total_input if total_input else 0.0

        return {
            "name": vname,
            "score_mean": float(np.mean(s)),
            "score_ci": float(ci),
            "score_std": float(np.std(s, ddof=1)) if n > 1 else 0.0,
            "latency_median": float(np.median(l)),
            "latency_p95": float(np.percentile(l, 95))
            if len(l) > 1
            else float(np.max(l)),
            "cost_per_query": float(np.mean(c)),
            "cost_monthly": float(np.mean(c)) * self.monthly_volume,
            "avg_cached_input_tokens": total_cached / len(ok_runs) if ok_runs else 0.0,
            "cache_hit_rate": cache_hit_rate,
            "total_runs": len(runs),
            "errors": errs,
            "failure_rate": errs / len(runs) if runs else 0,
        }

    def _score_band(self, score: float) -> tuple[str, str]:
        if score >= 9.0:
            return "excellent", "consistently delivers top-tier judged quality."
        if score >= 8.0:
            return "strong", "performs well and should satisfy most production use cases."
        if score >= 7.0:
            return "good", "is generally solid but leaves visible room for improvement."
        if score >= 6.0:
            return "usable", "can work, but quality tradeoffs will be noticeable."
        if score >= 5.0:
            return "weak", "is borderline acceptable and likely needs careful use."
        return "poor", "struggles on the current evaluation criteria."

    def _ci_band(self, score_mean: float, score_ci: float) -> tuple[str, str]:
        ratio = score_ci / max(score_mean, 0.1)
        if score_ci <= 0.35 or ratio <= 0.04:
            return "very stable", "repeated runs cluster tightly around the average."
        if score_ci <= 0.75 or ratio <= 0.08:
            return "stable", "the average looks dependable across repeated runs."
        if score_ci <= 1.25 or ratio <= 0.14:
            return "somewhat noisy", "the average is directionally useful but not razor precise."
        return "noisy", "results vary enough that the reported average should be treated cautiously."

    def _consistency_band(self, score_std: float) -> tuple[str, str]:
        if score_std <= 0.4:
            return "highly consistent", "scores barely move from run to run."
        if score_std <= 0.8:
            return "consistent", "run quality is fairly steady."
        if score_std <= 1.4:
            return "variable", "quality changes meaningfully between runs."
        return "erratic", "outputs swing a lot, so user experience may feel uneven."

    def _reliability_band(self, failure_rate: float) -> tuple[str, str]:
        if failure_rate == 0:
            return "fully reliable", "no execution failures were observed."
        if failure_rate <= 0.02:
            return "reliable", "failures are rare but worth monitoring at scale."
        if failure_rate <= 0.10:
            return "watchlist", "some failures appeared and could matter in production."
        if failure_rate <= 0.25:
            return "risky", "failure frequency is high enough to hurt trust."
        return "unstable", "too many runs failed for dependable deployment."

    def _latency_band(self, latency_median: float, latency_p95: float) -> tuple[str, str]:
        if latency_median <= 1000:
            speed = "very fast"
        elif latency_median <= 3000:
            speed = "fast"
        elif latency_median <= 8000:
            speed = "moderate"
        elif latency_median <= 15000:
            speed = "slow"
        else:
            speed = "very slow"

        tail_ratio = latency_p95 / max(latency_median, 1.0)
        if tail_ratio <= 1.2:
            tail = "tail latency is predictable."
        elif tail_ratio <= 1.8:
            tail = "worst-case latency is still reasonably controlled."
        elif tail_ratio <= 2.5:
            tail = "some slow spikes show up in the tail."
        else:
            tail = "tail latency is bursty, so users may occasionally wait much longer."

        return speed, tail

    def _cost_band(self, vname: str, cost_per_query: float) -> tuple[str, str]:
        ranked = sorted(
            ((name, self.summary(name)["cost_per_query"]) for name in self.by_variant),
            key=lambda item: item[1],
        )
        rank = next(i for i, (name, _) in enumerate(ranked) if name == vname)
        total = len(ranked)

        if cost_per_query <= 0.00001:
            return "free", "effectively free in this benchmark."
        if rank == 0:
            return "lowest cost", "is the cheapest option in the comparison."
        if rank < max(1, total // 3):
            return "low cost", "sits on the cheaper side of the pack."
        if rank >= max(1, total - max(1, total // 3)):
            return "premium cost", "is among the more expensive choices."
        return "mid-priced", "lands in the middle of the cost range."

    def metric_guide(self) -> list[dict]:
        """Short explanations of how to interpret each major metric."""
        return [
            {
                "metric": "Score",
                "meaning": "Higher is better. It is the judge's average quality rating on a 1-10 scale.",
            },
            {
                "metric": "±CI",
                "meaning": "Smaller is better. It shows uncertainty around the average score; wide intervals mean noisier results.",
            },
            {
                "metric": "Latency / P95",
                "meaning": "Median latency is the typical speed. P95 shows the slow end of the experience, which matters for user frustration.",
            },
            {
                "metric": "Std Dev / Failures",
                "meaning": "Lower standard deviation means more consistent quality. Lower failure rate means safer production behavior.",
            },
            {
                "metric": "p-value / Effect",
                "meaning": "A low p-value suggests the quality gap is probably real. Effect size says whether that gap is tiny or meaningful.",
            },
        ]

    def interpret_variant(self, vname: str) -> dict:
        """Human-readable interpretation for one variant."""
        summary = self.summary(vname)
        quality_label, quality_note = self._score_band(summary["score_mean"])
        ci_label, ci_note = self._ci_band(summary["score_mean"], summary["score_ci"])
        consistency_label, consistency_note = self._consistency_band(summary["score_std"])
        reliability_label, reliability_note = self._reliability_band(summary["failure_rate"])
        speed_label, speed_note = self._latency_band(
            summary["latency_median"], summary["latency_p95"]
        )
        cost_label, cost_note = self._cost_band(vname, summary["cost_per_query"])

        best_score = max(self.summary(name)["score_mean"] for name in self.by_variant)
        cheapest_cost = min(
            self.summary(name)["cost_per_query"] for name in self.by_variant
        )

        tradeoffs = []
        if summary["score_mean"] >= best_score - 0.3:
            tradeoffs.append("quality is near the top of the pack")
        if summary["cost_per_query"] <= cheapest_cost * 1.15:
            tradeoffs.append("cost is close to the cheapest option")
        if summary["failure_rate"] > 0.02:
            tradeoffs.append("reliability needs attention before scaling")
        if summary["latency_p95"] > summary["latency_median"] * 2:
            tradeoffs.append("tail latency may create occasional slow user experiences")

        takeaway = f"{vname} is a {quality_label} option with {cost_label} pricing."
        if tradeoffs:
            takeaway += " Key tradeoffs: " + "; ".join(tradeoffs) + "."

        return {
            "name": vname,
            "summary": summary,
            "quality_label": quality_label,
            "quality_note": quality_note,
            "ci_label": ci_label,
            "ci_note": ci_note,
            "consistency_label": consistency_label,
            "consistency_note": consistency_note,
            "reliability_label": reliability_label,
            "reliability_note": reliability_note,
            "speed_label": speed_label,
            "speed_note": speed_note,
            "cost_label": cost_label,
            "cost_note": cost_note,
            "takeaway": takeaway,
        }

    def compare(self, name_a: str, name_b: str) -> dict:
        """Statistical comparison between two variants."""
        a = self.scores[name_a]
        b = self.scores[name_b]

        # Welch's t-test
        if len(a) > 1 and len(b) > 1:
            t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
        else:
            t_stat, p_val = 0.0, 1.0

        # Cohen's d
        std_a = np.std(a, ddof=1) if len(a) > 1 else 0.001
        std_b = np.std(b, ddof=1) if len(b) > 1 else 0.001
        pooled = np.sqrt((std_a**2 + std_b**2) / 2)
        d = (float(np.mean(a)) - float(np.mean(b))) / pooled if pooled > 0 else 0.0

        # 95% CI of difference
        se = np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b))
        diff = float(np.mean(a) - np.mean(b))

        sig = p_val < 0.05
        winner = "none"
        if sig:
            winner = name_a if np.mean(a) > np.mean(b) else name_b

        abs_d = abs(d)
        mag = (
            "negligible"
            if abs_d < 0.2
            else "small" if abs_d < 0.5 else "medium" if abs_d < 0.8 else "large"
        )

        return {
            "a": name_a,
            "b": name_b,
            "mean_a": float(np.mean(a)),
            "mean_b": float(np.mean(b)),
            "p": float(p_val),
            "d": float(d),
            "ci_lower": diff - 1.96 * se,
            "ci_upper": diff + 1.96 * se,
            "significant": sig,
            "winner": winner,
            "magnitude": mag,
        }

    def interpret_comparison(self, name_a: str, name_b: str) -> dict:
        """Human-readable explanation of a pairwise comparison."""
        comparison = self.compare(name_a, name_b)
        diff = comparison["mean_a"] - comparison["mean_b"]
        direction = (
            f"{comparison['a']} leads by {abs(diff):.2f} points"
            if diff >= 0
            else f"{comparison['b']} leads by {abs(diff):.2f} points"
        )

        if comparison["significant"]:
            explanation = (
                f"{direction}, and the data suggests that this quality gap is likely real "
                f"(p={comparison['p']:.3f}, {comparison['magnitude']} effect)."
            )
        else:
            explanation = (
                f"{direction}, but the current sample does not show a reliable quality gap "
                f"(p={comparison['p']:.3f}). Treat these models as statistically tied for now."
            )

        return {
            **comparison,
            "direction": direction,
            "explanation": explanation,
        }

    def all_comparisons(self) -> list:
        names = [v.name for v in self.variants]
        return [self.compare(a, b) for a, b in itertools.combinations(names, 2)]

    def query_breakdown(self) -> list:
        rows = []
        for q_idx, query in enumerate(self.queries):
            row = {"query": _get_query_text(query)[:50]}
            for v in self.variants:
                v_scores = [
                    r.score
                    for r in self.by_variant[v.name]
                    if r.query_idx == q_idx and r.error is None
                ]
                row[v.name] = float(np.mean(v_scores)) if v_scores else 0.0
            rows.append(row)
        return rows

    def recommendation(self) -> dict:
        sums = {v.name: self.summary(v.name) for v in self.variants}

        best_name = max(sums, key=lambda v: sums[v]["score_mean"])

        costs_sorted = sorted(sums.items(), key=lambda x: x[1]["cost_per_query"])
        cheap_half = [v for v, _ in costs_sorted[: max(1, len(costs_sorted) // 2 + 1)]]
        value_name = max(cheap_half, key=lambda v: sums[v]["score_mean"])

        free = [v for v in sums if sums[v]["cost_per_query"] < 0.0001]
        free_name = max(free, key=lambda v: sums[v]["score_mean"]) if free else None

        return {
            "best": {"name": best_name, **sums[best_name]},
            "value": {"name": value_name, **sums[value_name]},
            "free": {"name": free_name, **sums[free_name]} if free_name else None,
        }

    def display(self):
        from verdict.display import print_report

        print_report(self)
