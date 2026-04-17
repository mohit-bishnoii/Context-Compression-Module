# evaluation/metrics.py
#
# Token counting helpers and results table printer.


def print_metrics_table(baseline_results: list, ccm_results: list):
    """Print a side-by-side comparison table to stdout."""
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


def _safe_avg(values: list) -> float:
    non_zero = [v for v in values if v > 0]
    return sum(non_zero) / len(non_zero) if non_zero else 0.0