# evaluation/run_evaluation.py
# Runs all 5 test conversations against both agents and reports results.

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from travel_agent.baseline_agent import BaselineAgent
from travel_agent.agent import CCMAgent
from evaluation.test_conversations import ALL_TESTS
from evaluation.assertions import check_response
from evaluation.metrics import print_metrics_table


def run_single_test(agent, test: dict, agent_type: str) -> dict:
    """Run one test conversation and return results."""
    print(f"\n{'='*60}")
    print(f"Running: {test['name']} [{agent_type}]")
    print(f"{'='*60}")

    agent.reset()
    turns = test["turns"]
    criteria = test["pass_criteria"]
    key_turn_index = criteria["turn_index"]

    responses = []
    tokens_per_turn = []

    for i, turn in enumerate(turns):
        user_msg = turn["content"]
        print(f"\nTurn {i+1}/{len(turns)}: {user_msg[:60]}...")

        try:
            result = agent.chat(user_msg)
            response = result["response"]
            tokens = result["tokens_in_context"]

            responses.append(response)
            tokens_per_turn.append(tokens)

            print(f"  Tokens: {tokens}")
            print(f"  Response: {response[:100]}...")

            # Small delay to avoid rate limits
            time.sleep(0.5)

        except Exception as e:
            print(f"  ERROR: {e}")
            responses.append(f"ERROR: {e}")
            tokens_per_turn.append(0)

    # Check the key turn response
    if key_turn_index < len(responses):
        key_response = responses[key_turn_index]
        assertion_result = check_response(key_response, criteria)
    else:
        assertion_result = {
            "passed": False,
            "score": 0.0,
            "details": "Key turn response not found"
        }

    print(f"\n--- RESULT: {test['name']} [{agent_type}] ---")
    print(f"Passed: {assertion_result['passed']}")
    print(f"Details: {assertion_result['details']}")
    print(f"Key turn response:\n{responses[key_turn_index] if key_turn_index < len(responses) else 'N/A'}")

    return {
        "test_name": test["name"],
        "agent_type": agent_type,
        "passed": assertion_result["passed"],
        "score": assertion_result["score"],
        "details": assertion_result["details"],
        "tokens_at_key_turn": tokens_per_turn[key_turn_index] if key_turn_index < len(tokens_per_turn) else 0,
        "tokens_per_turn": tokens_per_turn,
        "key_response": responses[key_turn_index] if key_turn_index < len(responses) else ""
    }


def run_full_evaluation(
    run_baseline: bool = True,
    tests_to_run: list = None
):
    """
    Run complete evaluation of both agents.

    Parameters:
      run_baseline: If False, skip baseline (saves time)
      tests_to_run: List of test indices to run (None = all)
    """
    tests = tests_to_run if tests_to_run else ALL_TESTS

    print("\n" + "="*60)
    print("STARTING FULL EVALUATION")
    print(f"Tests to run: {len(tests)}")
    print(f"Run baseline: {run_baseline}")
    print("="*60)

    baseline_results = []
    ccm_results = []

    # Initialize agents
    ccm_agent = CCMAgent(use_reranking=False)  # Faster for evaluation
    baseline_agent = BaselineAgent() if run_baseline else None

    for test in tests:
        # Run CCM agent
        ccm_result = run_single_test(ccm_agent, test, "CCM")
        ccm_results.append(ccm_result)

        # Run baseline agent
        if run_baseline and baseline_agent:
            baseline_result = run_single_test(
                baseline_agent, test, "Baseline"
            )
            baseline_results.append(baseline_result)
        else:
            baseline_results.append({
                "test_name": test["name"],
                "passed": False,
                "score": 0.0,
                "tokens_at_key_turn": 0,
                "details": "Baseline not run"
            })

    # Print results table
    print_metrics_table(baseline_results, ccm_results)

    return {
        "baseline": baseline_results,
        "ccm": ccm_results
    }


if __name__ == "__main__":
    # Run only CCM on Test A first to verify quickly
    results = run_full_evaluation(
        run_baseline=False,
        tests_to_run=ALL_TESTS[:1]  # Start with Test A only
    )
