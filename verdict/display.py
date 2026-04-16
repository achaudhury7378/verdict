"""
verdict/display.py — Terminal report.
"""

import numpy as np
from textwrap import wrap


def _print_wrapped(text: str, indent: str = "  ", width: int = 70):
    for line in wrap(text, width=max(20, width - len(indent))) or [""]:
        print(f"{indent}{line}")


def _tag(label: str, value: str) -> str:
    return f"[{label}: {value}]"


def print_report(a):
    """Print full verdict report."""

    W = 70
    names = [v.name for v in a.variants]

    # ── Header ──
    print(f"\n{'═' * W}")
    print(f"  verdict — {a.name}")
    n = len(a.variants)
    q = len(a.queries)
    r = len(a.results) // (n * q)
    print(f"  {n} variants × {q} queries × {r} runs = {len(a.results)} total")
    print(f"{'═' * W}")

    # ── Reading guide ──
    print(f"\n  METRIC CHEAT SHEET\n")
    for item in a.metric_guide():
        _print_wrapped(f"{item['metric']}: {item['meaning']}", indent="  ")

    # ── Results table ──
    sums = {v.name: a.summary(v.name) for v in a.variants}

    print(f"\n  RESULTS\n")
    print(
        f"  {'Variant':<16} {'Score':>7} {'±CI':>6} {'Latency':>9} "
        f"{'P95':>9} {'$/query':>9} {'$/month':>9}"
    )
    print(
        f"  {'─'*16} {'─'*7} {'─'*6} {'─'*9} "
        f"{'─'*9} {'─'*9} {'─'*9}"
    )

    for v in a.variants:
        s = sums[v.name]
        cost_s = f"${s['cost_per_query']:.4f}" if s["cost_per_query"] > 0.00001 else "FREE"
        month_s = f"${s['cost_monthly']:.0f}" if s["cost_monthly"] > 0.01 else "$0"
        print(
            f"  {v.name:<16} {s['score_mean']:>5.1f}/10 {s['score_ci']:>5.1f} "
            f"{s['latency_median']:>7,.0f}ms {s['latency_p95']:>7,.0f}ms "
            f"{cost_s:>9} {month_s:>9}"
        )

    # ── Consistency ──
    print(f"\n  CONSISTENCY\n")
    print(f"  {'Variant':<16} {'Std Dev':>8} {'Failures':>10}")
    print(f"  {'─'*16} {'─'*8} {'─'*10}")
    for v in a.variants:
        s = sums[v.name]
        f_str = f"{s['failure_rate']*100:.0f}%" if s["failure_rate"] > 0 else "0%"
        print(f"  {v.name:<16} {s['score_std']:>7.2f} {f_str:>10}")

    # ── Interpretation ──
    print(f"\n{'═' * W}")
    print(f"  QUICK READ\n")
    for v in a.variants:
        interp = a.interpret_variant(v.name)
        summary = interp["summary"]
        cache_note = ""
        if summary.get("cache_hit_rate", 0) > 0:
            cache_note = f" | Cache hit {summary['cache_hit_rate']*100:.0f}%"
        print(f"  {v.name}")
        _print_wrapped(
            " ".join(
                [
                    _tag("Quality", interp["quality_label"]),
                    _tag("Confidence", interp["ci_label"]),
                    _tag("Consistency", interp["consistency_label"]),
                ]
            ),
            indent="    ",
            width=W,
        )
        _print_wrapped(
            " ".join(
                [
                    _tag("Reliability", interp["reliability_label"]),
                    _tag("Speed", interp["speed_label"]),
                    _tag("Cost", interp["cost_label"]),
                ]
            ),
            indent="    ",
            width=W,
        )
        _print_wrapped(
            f"Score {summary['score_mean']:.1f}/10 (+/- {summary['score_ci']:.2f}) | "
            f"Median {summary['latency_median']:.0f}ms | "
            f"P95 {summary['latency_p95']:.0f}ms | "
            f"Failures {summary['failure_rate']*100:.0f}%{cache_note}",
            indent="    ",
            width=W,
        )
        _print_wrapped(
            f"Takeaway: {interp['takeaway']}",
            indent="    ",
            width=W,
        )
        print()

    # ── Stats ──
    comps = a.all_comparisons()
    if comps:
        print(f"\n{'═' * W}")
        print(f"  STATISTICAL COMPARISONS\n")
        print(
            f"  {'Matchup':<32} {'p-value':>8} {'Effect':>10} {'Verdict':>22}"
        )
        print(f"  {'─'*32} {'─'*8} {'─'*10} {'─'*22}")

        for c in comps:
            p_s = f"{c['p']:.3f}" if c["p"] >= 0.001 else "<0.001"
            d_s = f"d={abs(c['d']):.2f}"
            if c["significant"]:
                v_s = f"{c['winner']} ({c['magnitude']})"
            else:
                v_s = "No sig. diff"
            m = f"{c['a']} vs {c['b']}"
            print(f"  {m:<32} {p_s:>8} {d_s:>10} {v_s:>22}")

        print(f"\n  STATS IN PLAIN ENGLISH\n")
        for c in comps:
            interp = a.interpret_comparison(c["a"], c["b"])
            _print_wrapped(interp["explanation"], indent="  ", width=W)

    # ── Per-query ──
    bd = a.query_breakdown()
    if bd:
        print(f"\n{'═' * W}")
        print(f"  PER-QUERY BREAKDOWN\n")

        header = f"  {'Query':<32}"
        for v in a.variants:
            header += f" {v.name:>10}"
        print(header)

        sep = f"  {'─'*32}"
        for v in a.variants:
            sep += f" {'─'*10}"
        print(sep)

        for row in bd:
            line = f"  {row['query']:<32}"
            for v in a.variants:
                line += f" {row.get(v.name, 0):>8.1f}/10"
            print(line)

    # ── Recommendation ──
    rec = a.recommendation()
    best = rec["best"]
    value = rec["value"]
    free = rec.get("free")

    print(f"\n{'═' * W}")
    print(f"  ★ RECOMMENDATION\n")
    print(f"  BEST OVERALL:  {best['name']} ({best['score_mean']:.1f} ± {best['score_ci']:.1f})")

    if value["name"] != best["name"]:
        q_pct = value["score_mean"] / best["score_mean"] * 100
        save = best["cost_monthly"] - value["cost_monthly"]
        print(
            f"  BEST VALUE:    {value['name']} — "
            f"{q_pct:.0f}% quality, saves ${save:,.0f}/mo"
        )
    else:
        print(f"  BEST VALUE:    {value['name']} (same as best)")

    if free and free["name"] and free["name"] != best["name"]:
        f_note = (
            f", {free['failure_rate']*100:.0f}% failures"
            if free["failure_rate"] > 0.01
            else ""
        )
        print(f"  BEST FREE:     {free['name']} ({free['score_mean']:.1f}/10{f_note})")

    # ── Insight ──
    sorted_v = sorted(names, key=lambda v: np.mean(a.scores[v]), reverse=True)
    if len(sorted_v) >= 2:
        for c in comps:
            if set([c["a"], c["b"]]) == set(sorted_v[:2]):
                if not c["significant"]:
                    cheaper = (
                        c["a"]
                        if sums[c["a"]]["cost_per_query"]
                        <= sums[c["b"]]["cost_per_query"]
                        else c["b"]
                    )
                    print(
                        f"\n  💡 {c['a']} and {c['b']} are statistically "
                        f"tied (p={c['p']:.3f})."
                    )
                    print(f"     Pick {cheaper} — same quality, lower cost.")
                else:
                    print(
                        f"\n  💡 {c['winner']} is significantly better "
                        f"(p={c['p']:.3f}, {c['magnitude']} effect)."
                    )
                break

    print(f"\n{'═' * W}\n")
