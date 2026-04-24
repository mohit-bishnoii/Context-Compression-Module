# evaluation/metrics.py
#
# Hackathon metrics: token reduction, latency, cost, task success rate,
# factual recall, coherence, tool-call correctness, multi-session
# continuity, and omission / distortion rate.
#
# Groq llama-3.3-70b-versatile pricing (as of 2025):
#   Input:  $0.59 / 1M tokens
#   Output: $0.79 / 1M tokens
# We approximate output as ~25% of total context tokens.

GROQ_INPUT_COST_PER_1M  = 0.59   # USD
GROQ_OUTPUT_COST_PER_1M = 0.79   # USD
OUTPUT_FRACTION         = 0.25   # estimated output / total ratio


# ─────────────────────────────────────────────────────────────────
# Cost helpers
# ─────────────────────────────────────────────────────────────────

def compute_cost(total_tokens: int) -> float:
    """
    Estimate USD cost for a conversation given total context tokens.

    Uses blended input/output split (75% input, 25% output).
    """
    input_tok  = total_tokens * (1 - OUTPUT_FRACTION)
    output_tok = total_tokens * OUTPUT_FRACTION
    return (
        input_tok  / 1_000_000 * GROQ_INPUT_COST_PER_1M +
        output_tok / 1_000_000 * GROQ_OUTPUT_COST_PER_1M
    )


def pct_reduction(baseline: float, ccm: float) -> float:
    """Percentage reduction: how much smaller is ccm vs baseline."""
    if baseline <= 0:
        return 0.0
    return round((baseline - ccm) / baseline * 100, 1)


# ─────────────────────────────────────────────────────────────────
# Coherence heuristic
# ─────────────────────────────────────────────────────────────────

# Pairs: if responses contain the first word they MUST also contain
# the second (consistency check).
_COHERENCE_PAIRS = [
    ("shellfish", "allerg"),
    ("allerg",    "avoid"),
    ("budget",    "$"),
    ("switzerland", "zurich"),
]

_CONTRADICTION_PAIRS = [
    # (old_term, new_term) — if both appear together it is suspicious
    ("bali",    "switzerland"),
    ("beach",   "mountains"),
]


def score_coherence(responses: list[str]) -> dict:
    """
    Semi-quantitative coherence score over all turns.

    Checks:
      1. No contradictory place names co-occur across responses.
      2. Critical constraint words, once introduced, persist.

    Returns
    -------
    dict with keys: score (0-1), issues (list[str])
    """
    full_text = " ".join(r.lower() for r in responses if r)
    issues = []

    # Contradiction check
    for old, new in _CONTRADICTION_PAIRS:
        if old in full_text and new in full_text:
            issues.append(f"Possible contradiction: '{old}' and '{new}' both appear")

    # Consistency check — once introduced, stays
    for anchor, companion in _COHERENCE_PAIRS:
        anchor_turns = [
            i for i, r in enumerate(responses)
            if anchor in r.lower()
        ]
        if anchor_turns:
            last = anchor_turns[-1]
            late_responses = " ".join(r.lower() for r in responses[last:])
            if anchor in late_responses and companion not in late_responses:
                issues.append(
                    f"Coherence gap: '{anchor}' present but '{companion}' missing "
                    f"in later turns"
                )

    score = max(0.0, 1.0 - len(issues) * 0.2)
    return {"score": round(score, 2), "issues": issues}


# ─────────────────────────────────────────────────────────────────
# Factual recall scorer
# ─────────────────────────────────────────────────────────────────

def score_factual_recall(response: str, facts_to_check: list[str]) -> dict:
    """
    Score how many planted facts are mentioned in a response.

    Parameters
    ----------
    response       : str        The agent's response text
    facts_to_check : list[str]  Keywords / phrases to look for

    Returns
    -------
    dict  {recalled: int, total: int, score: float, missing: list[str]}
    """
    if not facts_to_check:
        return {"recalled": 0, "total": 0, "score": 1.0, "missing": []}

    low = response.lower()
    recalled = [f for f in facts_to_check if f.lower() in low]
    missing  = [f for f in facts_to_check if f.lower() not in low]
    score    = len(recalled) / len(facts_to_check)
    return {
        "recalled": len(recalled),
        "total":    len(facts_to_check),
        "score":    round(score, 2),
        "missing":  missing,
    }


# ─────────────────────────────────────────────────────────────────
# Tool-call correctness scorer
# ─────────────────────────────────────────────────────────────────

def score_tool_calls(
    actual_tool_calls: list[dict],
    expected_tools: list[str],
) -> dict:
    """
    Score whether the right tools were called.

    Parameters
    ----------
    actual_tool_calls : list[dict]  Each dict has key "tool"
    expected_tools    : list[str]   Tool names that SHOULD have been called

    Returns
    -------
    dict  {correct: int, total_expected: int, score: float,
           extra_calls: list, missing_calls: list}
    """
    if not expected_tools:
        return {
            "correct": 0, "total_expected": 0, "score": 1.0,
            "extra_calls": [], "missing_calls": []
        }

    called = [c.get("tool", "") for c in actual_tool_calls]
    correct = [t for t in expected_tools if t in called]
    missing = [t for t in expected_tools if t not in called]
    extra   = [t for t in called if t not in expected_tools]
    score   = len(correct) / len(expected_tools)
    return {
        "correct":        len(correct),
        "total_expected": len(expected_tools),
        "score":          round(score, 2),
        "missing_calls":  missing,
        "extra_calls":    extra,
    }


# ─────────────────────────────────────────────────────────────────
# Omission rate scorer
# ─────────────────────────────────────────────────────────────────

def score_omission_rate(
    key_fields_preserved: int,
    key_fields_total: int,
) -> dict:
    """
    Measure what fraction of key raw fields survived compression.

    Inputs come from ToolCompressor.get_omission_stats().

    Returns
    -------
    dict  {preserved: int, total: int, omission_rate: float}
    """
    if key_fields_total == 0:
        return {"preserved": 0, "total": 0, "omission_rate": 0.0}
    omission = 1.0 - key_fields_preserved / key_fields_total
    return {
        "preserved":     key_fields_preserved,
        "total":         key_fields_total,
        "omission_rate": round(omission, 3),
    }


# ─────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────

def _safe_avg(values: list) -> float:
    non_zero = [v for v in values if v > 0]
    return sum(non_zero) / len(non_zero) if non_zero else 0.0


def _fmt(val, decimals=1, suffix=""):
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{decimals}f}{suffix}"
    return f"{val}{suffix}"


# ─────────────────────────────────────────────────────────────────
# Main summary printer — all 9 hackathon metrics
# ─────────────────────────────────────────────────────────────────

def print_metrics_table(baseline_results: list, ccm_results: list):
    """Print the legacy pass/fail side-by-side table (kept for compatibility)."""
    print("\n" + "=" * 72)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 72)

    header = (
        f"{'Test':<36} {'Baseline':^12} {'CCM':^12} {'Tok@key (CCM)':>12}"
    )
    print(header)
    print("-" * 72)

    for b, c in zip(baseline_results, ccm_results):
        b_str = "✅ PASS" if b.get("passed") else "❌ FAIL"
        c_str = "✅ PASS" if c.get("passed") else "❌ FAIL"
        tok   = c.get("tokens_at_key_turn", 0)
        name  = c["test_name"][:34]
        print(f"{name:<36} {b_str:^12} {c_str:^12} {tok:>12}")

    print("-" * 72)

    b_pass = sum(1 for r in baseline_results if r.get("passed"))
    c_pass = sum(1 for r in ccm_results     if r.get("passed"))
    n      = len(ccm_results)
    print(f"{'TOTAL':<36} {b_pass}/{n:^10}  {c_pass}/{n:^10}")

    if baseline_results and ccm_results:
        b_avg = _safe_avg(
            [r.get("tokens_at_key_turn", 0) for r in baseline_results]
        )
        c_avg = _safe_avg(
            [r.get("tokens_at_key_turn", 0) for r in ccm_results]
        )
        ratio = round(b_avg / max(c_avg, 1), 1)
        print(
            f"\nAvg tokens @ key turn — Baseline: {b_avg:.0f}  "
            f"CCM: {c_avg:.0f}  Ratio: {ratio}x"
        )

    print("=" * 72 + "\n")


def print_hackathon_metrics(
    baseline_results: list,
    ccm_results: list,
    global_stats: dict = None,
):
    """
    Print the full hackathon metric card covering all 9 required dimensions.

    Parameters
    ----------
    baseline_results : list[dict]   Per-test result dicts from baseline runs
    ccm_results      : list[dict]   Per-test result dicts from CCM runs
    global_stats     : dict         Aggregated stats (latency, omission, etc.)
                                    Populated by run_evaluation.py
    """
    gs = global_stats or {}

    # ── Aggregate basic numbers ──────────────────────────────────
    n = len(ccm_results)

    b_tokens  = [r.get("tokens_at_key_turn", 0) for r in baseline_results]
    c_tokens  = [r.get("tokens_at_key_turn", 0) for r in ccm_results]
    b_avg_tok = _safe_avg(b_tokens)
    c_avg_tok = _safe_avg(c_tokens)
    tok_ratio = round(b_avg_tok / max(c_avg_tok, 1), 1)

    b_total_tok = gs.get("baseline_total_tokens", sum(b_tokens))
    c_total_tok = gs.get("ccm_total_tokens",      sum(c_tokens))
    b_cost = compute_cost(b_total_tok)
    c_cost = compute_cost(c_total_tok)

    b_pass = sum(1 for r in baseline_results if r.get("passed"))
    c_pass = sum(1 for r in ccm_results      if r.get("passed"))
    b_success_rate = b_pass / n if n else 0
    c_success_rate = c_pass / n if n else 0

    b_latency = gs.get("baseline_avg_latency_s", None)
    c_latency = gs.get("ccm_avg_latency_s",      None)

    # Factual recall
    b_recall = _safe_avg([r.get("factual_recall_score", 0) for r in baseline_results])
    c_recall = _safe_avg([r.get("factual_recall_score", 0) for r in ccm_results])

    # Coherence
    b_coherence = _safe_avg([r.get("coherence_score", 0) for r in baseline_results])
    c_coherence = _safe_avg([r.get("coherence_score", 0) for r in ccm_results])

    # Tool-call correctness
    b_tool = _safe_avg([r.get("tool_call_correctness", 0) for r in baseline_results])
    c_tool = _safe_avg([r.get("tool_call_correctness", 0) for r in ccm_results])

    # Multi-session continuity (boolean flag from test 7)
    ms_continuity = gs.get("multi_session_continuity", None)

    # Omission rate
    omission = gs.get("omission_rate", None)
    comp_ratio = gs.get("overall_compression_ratio", tok_ratio)

    # ── Print card ───────────────────────────────────────────────
    W = 65
    print("\n" + "═" * W)
    print("  CCM HACKATHON EVALUATION — ALL 9 REQUIRED METRICS")
    print("═" * W)
    print(f"  {'Metric':<38} {'Baseline':>10}  {'CCM':>10}")
    print("─" * W)

    def row(label, b_val, c_val, highlight=False):
        star = " ◀" if highlight else ""
        print(f"  {label:<38} {b_val:>10}  {c_val:>10}{star}")

    # 1. Token reduction / compression ratio
    row("1. Avg tokens/turn (context)",
        _fmt(b_avg_tok, 0),
        _fmt(c_avg_tok, 0))
    row("   Compression ratio (CCM vs Baseline)",
        "—",
        _fmt(comp_ratio, 1, "x"),
        highlight=True)
    row("   Token reduction %",
        "—",
        _fmt(pct_reduction(b_avg_tok, c_avg_tok), 1, "%"),
        highlight=True)

    # 2. Latency reduction
    if b_latency and c_latency:
        lat_pct = pct_reduction(b_latency, c_latency)
        row("2. Avg latency / turn (s)",
            _fmt(b_latency, 2, "s"),
            _fmt(c_latency, 2, "s"))
        row("   Latency reduction %",
            "—",
            _fmt(lat_pct, 1, "%"),
            highlight=True)
    else:
        row("2. Avg latency / turn (s)",
            "N/A", "N/A")

    # 3. Cost reduction
    row("3. Est. cost / session (USD)",
        f"${b_cost:.4f}",
        f"${c_cost:.4f}")
    row("   Cost reduction %",
        "—",
        _fmt(pct_reduction(b_cost, c_cost), 1, "%"),
        highlight=True)

    # 4. Task success rate
    row("4. Task success rate",
        f"{b_pass}/{n} ({b_success_rate:.0%})",
        f"{c_pass}/{n} ({c_success_rate:.0%})",
        highlight=True)

    # 5. Factual retention / recall
    row("5. Factual recall score",
        _fmt(b_recall, 2),
        _fmt(c_recall, 2),
        highlight=True)

    # 6. Coherence over long turns
    row("6. Coherence score (0–1)",
        _fmt(b_coherence, 2) if b_coherence else "N/A",
        _fmt(c_coherence, 2) if c_coherence else "N/A")

    # 7. Tool-call correctness
    row("7. Tool-call correctness",
        _fmt(b_tool, 2) if b_tool else "N/A",
        _fmt(c_tool, 2) if c_tool else "N/A")

    # 8. Multi-session continuity
    if ms_continuity is not None:
        ms_str = "✅ PASS" if ms_continuity else "❌ FAIL"
        row("8. Multi-session continuity", "—", ms_str)
    else:
        row("8. Multi-session continuity", "—", "not tested")

    # 9. Omission / distortion rate
    if omission is not None:
        row("9. Omission rate (compression)",
            "—",
            _fmt(omission * 100, 1, "%"))
    else:
        row("9. Omission rate (compression)", "—", "N/A")

    print("═" * W)
    print(f"  Tests passed: Baseline {b_pass}/{n}  |  CCM {c_pass}/{n}")
    if c_total_tok and b_total_tok:
        saved_tokens = b_total_tok - c_total_tok
        saved_cost   = b_cost - c_cost
        print(f"  Total tokens saved: {saved_tokens:,}")
        print(f"  Estimated cost saved: ${saved_cost:.4f}")
    print("═" * W + "\n")