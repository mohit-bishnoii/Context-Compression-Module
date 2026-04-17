# evaluation/assertions.py
#
# Pass/fail assertion functions for the 5 evaluation conversations.
# Used by run_evaluation.py to check agent responses automatically.


def check_response(response: str, criteria: dict) -> dict:
    """
    Check an agent response against pass criteria.

    Supported criteria keys
    -----------------------
    must_contain_any      list[str]  At least one must appear in response
    must_not_contain_any  list[str]  None may appear in response
    must_not_contain      list[str]  Alias for must_not_contain_any
    case_sensitive        bool       Default False

    Returns
    -------
    dict  {passed: bool, score: float, details: str}
    """
    if not response:
        return {"passed": False, "score": 0.0, "details": "Empty response"}

    case_sensitive = criteria.get("case_sensitive", False)

    if case_sensitive:
        check_text = response
        norm = lambda w: w
    else:
        check_text = response.lower()
        norm = lambda w: w.lower()

    details = []
    passed  = True

    # must_contain_any ────────────────────────────────────────────
    must_contain = criteria.get("must_contain_any", [])
    if must_contain:
        hits = [w for w in must_contain if norm(w) in check_text]
        if hits:
            details.append(f"PASS: found required term(s): {hits}")
        else:
            passed = False
            details.append(
                f"FAIL: none of required terms found: {must_contain}"
            )

    # must_not_contain_any / must_not_contain ─────────────────────
    forbidden = list(criteria.get("must_not_contain_any", []))
    forbidden += list(criteria.get("must_not_contain",     []))
    if forbidden:
        hits = [w for w in forbidden if norm(w) in check_text]
        if hits:
            passed = False
            details.append(f"FAIL: forbidden term(s) found: {hits}")
        else:
            details.append("PASS: no forbidden terms found")

    score = 1.0 if passed else 0.0
    return {
        "passed":  passed,
        "score":   score,
        "details": " | ".join(details) if details else "no criteria defined",
    }