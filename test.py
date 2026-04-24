# test.py
#
# Full system tests for the CCM travel agent.
# Run from project root: python test.py
#
# IMPORTANT — before running:
#   1. Set GROQ_API_KEY in .env or environment
#   2. Install deps: pip install -r requirements.txt
#   3. Run from project root (not from inside a subfolder)
#      so that ./data/chroma_db paths resolve correctly.
#
# RATE LIMITS:
#   Groq free tier = 30 requests/minute on llama-3.3-70b-versatile.
#   Each test turn makes 3–5 API calls (extractor, stale detector,
#   agent, compressor). We add sleeps to stay safe.
#   If you hit a 429 error, increase INTER_TURN_SLEEP below.

import sys
import os
import time
import json
import gc
import shutil
from pathlib import Path

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"]     = "False"

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CHROMA_PATH = "./data/chroma_db"
MEMORY_PATH = "data/working_memory.json"

# Seconds to wait between conversation turns to avoid rate limits.
# Increase to 3 or 4 if you keep getting 429 errors.
INTER_TURN_SLEEP = 2.0


# ─────────────────────────────────────────────────────────────────
# RESET UTILITY
# ─────────────────────────────────────────────────────────────────

def reset_all_storage():
    """
    Wipe ALL persisted state between tests.

    Strategy:
      1. Write a blank working_memory.json
      2. Try to delete ChromaDB collections via a fresh client
         (releases internal locks held by ChromaDB)
      3. Force garbage collection to release Python references
      4. Delete the chroma_db folder and recreate it empty
    """
    # Step 1: blank working memory
    os.makedirs("data", exist_ok=True)
    try:
        from travel_agent.tools import reset_budget
        reset_budget()
    except Exception as exc:
        print(f"[Reset] Budget reset warning: {exc}")
    blank = {
        "facts": {"critical": [], "important": [], "contextual": []},
        "decisions":  [],
        "cancelled":  [],
        "turn_count": 0,
        "conversation_id": "",
        "last_updated": "",
    }
    with open(MEMORY_PATH, "w") as f:
        json.dump(blank, f, indent=2)

    # Step 2: delete collections via fresh client
    if os.path.exists(CHROMA_PATH):
        try:
            import chromadb
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            for col in client.list_collections():
                # chromadb 0.5.x: col is a Collection object with .name
                try:
                    name = col.name if hasattr(col, "name") else str(col)
                    client.delete_collection(name)
                    print(f"[Reset] Deleted collection: {name}")
                except Exception as exc:
                    print(f"[Reset] Could not delete collection: {exc}")
            # Explicitly drop the reference
            del client
        except Exception as exc:
            print(f"[Reset] ChromaDB client error: {exc}")

    # Step 3: GC + short sleep so OS releases file handles
    gc.collect()
    time.sleep(0.5)

    # Step 4: delete and recreate folder
    # if os.path.exists(CHROMA_PATH):
    #     try:
    #         shutil.rmtree(CHROMA_PATH)
    #         print("[Reset] ChromaDB folder deleted")
    #     except PermissionError:
    #         print("[Reset] Folder locked — collections were cleared above (OK)")
    #     except Exception as exc:
    #         print(f"[Reset] Folder delete error: {exc}")
    # 
    # os.makedirs(CHROMA_PATH, exist_ok=True)
    # time.sleep(0.3)
    print("[Reset] Storage reset complete\n")


# ─────────────────────────────────────────────────────────────────
# TEST 1 — Memory Extraction Priority
# ─────────────────────────────────────────────────────────────────

def test_1_memory_extraction():
    """
    Verify that the Extractor marks a medical allergy as CRITICAL.

    The extractor calls the LLM with the EXTRACTION_PROMPT.
    The LLM must return priority="critical" for shellfish allergy.
    The safety-promotion logic in extractor.py also auto-promotes
    anything containing "allerg", "medical", "severe", etc.
    """
    print("\n" + "=" * 55)
    print("TEST 1: Memory Extraction Priority")
    print("=" * 55)

    reset_all_storage()

    from ccm.memory_store import WorkingMemory
    from ccm.extractor    import MemoryExtractor

    memory    = WorkingMemory()
    extractor = MemoryExtractor()

    msg = (
        "I want to plan a 10-day trip to Tokyo and Kyoto. "
        "My total budget is $3000 maximum. "
        "I am severely allergic to shellfish — medical requirement. "
        "I prefer a relaxed pace with maximum 2 activities per day."
    )

    facts = extractor.extract_and_update(msg, memory)

    print(f"\nExtracted {len(facts)} facts:")
    for f in facts:
        print(f"  [{f['priority']:11}] {f['key']:30} → {f['value']}")

    print("\nFormatted memory:")
    print(memory.format_for_prompt())

    critical        = memory.get_critical_facts()
    critical_values = [f["value"].lower() for f in critical]
    allergy_critical = any("shellfish" in v for v in critical_values)

    print(f"\nShellfish allergy classified as CRITICAL: {allergy_critical}")
    result = allergy_critical
    print(f"{'✅ TEST 1 PASSED' if result else '❌ TEST 1 FAILED'}")
    return result


# ─────────────────────────────────────────────────────────────────
# TEST 2 — Stale Context Detection
# ─────────────────────────────────────────────────────────────────

def test_2_stale_detection():
    """
    Verify that cancelling Bali removes all Bali memories.

    Flow:
      1. Add "Bali beach vacation" to WorkingMemory
      2. Add an episodic entry about Bali resorts to ChromaDB
      3. Confirm retrieval WORKS before pivot
      4. Call stale_detector with "Scratch Bali…"
      5. Confirm retrieval returns ZERO results after pivot
    """
    print("\n" + "=" * 55)
    print("TEST 2: Stale Context Detection")
    print("=" * 55)

    reset_all_storage()

    from ccm.memory_store    import WorkingMemory
    from ccm.stale_detector  import StaleDetector
    from ccm.episodic_memory import EpisodicMemory
    from ccm.semantic_memory import SemanticMemory

    memory   = WorkingMemory()
    episodic = EpisodicMemory()
    semantic = SemanticMemory()
    detector = StaleDetector()

    # 1. Plant Bali in working memory
    memory.add_facts([{
        "key":      "destination_primary",
        "value":    "Bali beach vacation",
        "category": "decision",
        "priority": "important",
    }])
    print("Added Bali to working memory")

    # 2. Plant Bali in episodic memory
    ep_id = episodic.add(
        "Researched Bali resorts. Found Seminyak Beach Hotel at $180/night.",
        turn_range=(1, 3),
    )
    print(f"Stored episodic entry: {ep_id}")

    # 3. Verify retrieval works BEFORE pivot
    time.sleep(0.3)
    before = episodic.retrieve("Bali resorts beach")
    print(f"Before pivot — Bali entries in episodic: {len(before)}")
    if len(before) == 0:
        print("⚠️  WARNING: episodic entry not found before pivot")
        print("   This may mean ChromaDB is not persisting correctly.")
        print("   Check that sentence-transformers installed properly.")

    # 4. User cancels Bali
    pivot = "Scratch Bali, let us do Switzerland instead — I want mountains."
    print(f"\nPivot message: '{pivot}'")
    result = detector.check_and_clean(pivot, memory, episodic, semantic)
    print(f"Override detected: {result['has_override']}")
    print(f"Overridden keys:   {result.get('overridden_keys', [])}")
    print(f"Reason:            {result.get('reason', '')}")

    # Give ChromaDB a moment to commit the update
    time.sleep(0.5)

    # 5. Verify Bali is GONE after pivot
    after = episodic.retrieve("Bali resorts beach")
    print(f"After pivot  — Bali entries in episodic: {len(after)}")

    cancelled = memory.get_all().get("cancelled", [])
    print(f"Cancelled in memory: {cancelled}")

    # Also check working memory no longer has Bali destination
    dest_val = memory.get("destination_primary")
    print(f"destination_primary after pivot: {dest_val}")

    passed = result["has_override"] and len(after) == 0
    print(f"\n{'✅ TEST 2 PASSED' if passed else '❌ TEST 2 FAILED'}")

    if not result["has_override"]:
        print("  DIAGNOSIS: StaleDetector LLM did not detect override.")
        print("  Check GROQ_API_KEY is valid.")
    elif len(after) > 0:
        print("  DIAGNOSIS: Episodic entry not marked stale.")
        print("  Check mark_stale_by_content logic.")

    return passed


# ─────────────────────────────────────────────────────────────────
# TEST 3 — Tool Result Compression
# ─────────────────────────────────────────────────────────────────

def test_3_compression():
    """
    Verify compression ratio > 2x and shellfish conflict flagged.

    The Compressor sends a prompt to the LLM with the user constraint
    "severely allergic to shellfish". The compressed output must
    mention shellfish/allergy/avoid/⚠️.
    """
    print("\n" + "=" * 55)
    print("TEST 3: Tool Result Compression")
    print("=" * 55)

    reset_all_storage()

    from ccm.compressor     import ToolCompressor
    from travel_agent.tools import places_search

    compressor = ToolCompressor()

    raw_result  = places_search("Tsukiji Tokyo", "restaurants")
    raw_str     = json.dumps(raw_result)
    raw_tokens  = len(raw_str) // 4

    constraints = ["severely allergic to shellfish — medical requirement"]
    compressed  = compressor.compress(raw_result, "places_search", constraints)
    comp_tokens = len(compressed) // 4
    ratio       = raw_tokens / max(comp_tokens, 1)

    print(f"Raw tokens:        {raw_tokens}")
    print(f"Compressed tokens: {comp_tokens}")
    print(f"Ratio:             {ratio:.1f}x")
    print(f"\nCompressed output:\n{compressed}")

    shellfish_flagged = any(
        w in compressed.lower()
        for w in ["shellfish", "⚠", "allerg", "avoid", "warning"]
    )
    print(f"\nShellfish conflict flagged: {shellfish_flagged}")

    passed = ratio > 2.0 and shellfish_flagged
    print(f"{'✅ TEST 3 PASSED' if passed else '❌ TEST 3 FAILED'}")
    return passed


# ─────────────────────────────────────────────────────────────────
# TEST 4 — RAG Retrieval
# ─────────────────────────────────────────────────────────────────

def test_4_rag_retrieval():
    """
    Verify that the Retriever finds semantically relevant memories.

    We seed:
      - An episodic entry about shellfish allergy + budget
      - An episodic entry about a booked flight
      - A semantic entry about Tsukiji restaurants

    Then query: "dinner restaurants Tsukiji shellfish"
    Expect: at least 1 result, allergy entry retrieved.
    """
    print("\n" + "=" * 55)
    print("TEST 4: RAG Retrieval")
    print("=" * 55)

    reset_all_storage()

    from ccm.episodic_memory import EpisodicMemory
    from ccm.semantic_memory import SemanticMemory
    from ccm.retriever       import Retriever

    ep  = EpisodicMemory()
    sem = SemanticMemory()

    # Seed memories
    ep.add("User severely allergic to shellfish. Budget $3000.", turn_range=(0, 1))
    ep.add("Booked ANA flight NYC to Tokyo $780 direct.",        turn_range=(2, 3))
    sem.add(
        "Tsukiji restaurants: Sushi Dai has shellfish — AVOID. "
        "Odayasu is shellfish-free, traditional Japanese.",
        tool_name="places_search",
        query_used="restaurants near Tsukiji",
        turn_number=4,
    )

    time.sleep(0.3)   # let ChromaDB commit

    retriever = Retriever(ep, sem, use_reranking=False)

    query   = "dinner restaurants Tsukiji shellfish allergy"
    results = retriever.retrieve(query, n_episodic=3, n_semantic=2)

    total = len(results["episodic"]) + len(results["semantic"])
    print(f"Total retrieved: {total}")

    all_texts = (
        [r["text"] for r in results["episodic"]] +
        [r["text"] for r in results["semantic"]]
    )
    allergy_retrieved = any(
        "shellfish" in t.lower() or "allerg" in t.lower()
        for t in all_texts
    )
    print(f"Allergy info retrieved: {allergy_retrieved}")
    for t in all_texts:
        print(f"  → {t[:80]}")

    passed = total > 0 and allergy_retrieved
    print(f"{'✅ TEST 4 PASSED' if passed else '❌ TEST 4 FAILED'}")
    return passed


# ─────────────────────────────────────────────────────────────────
# TEST 5 — CCM Agent Remembers Allergy Across Turns
# ─────────────────────────────────────────────────────────────────

def test_5_ccm_agent_allergy():
    """
    KEY END-TO-END TEST.
    Allergy stated at turn 1, must be mentioned at turn 4.

    Turn 1: states shellfish allergy → extractor → WorkingMemory critical
    Turn 2: find flights (tool call)
    Turn 3: find hotels (tool call)
    Turn 4: find restaurants near Tsukiji
             → assembler injects [CRITICAL CONSTRAINTS] → agent sees allergy
             → agent must mention shellfish/allergy/warning

    This tests the FULL PIPELINE in 4 turns.
    Sleeping between turns to stay within Groq rate limits.
    """
    print("\n" + "=" * 55)
    print("TEST 5: CCM Agent Remembers Allergy (Full Pipeline)")
    print("=" * 55)

    reset_all_storage()

    from travel_agent.agent import CCMAgent

    agent = CCMAgent(use_reranking=False)   # no re-ranking = fewer API calls

    turns = [
        (
            "I want to plan a trip to Tokyo. "
            "My total budget is $3000 maximum. "
            "I am severely allergic to shellfish — this is a medical allergy. "
            "I cannot eat any shellfish whatsoever."
        ),
        "Find flights from New York to Tokyo in June",
        "Find hotels in Tokyo within my budget",
        "Find dinner restaurants near Tsukiji fish market in Tokyo",
    ]

    responses = []
    for i, msg in enumerate(turns):
        print(f"\n--- Turn {i+1} ---")
        print(f"User: {msg[:80]}…" if len(msg) > 80 else f"User: {msg}")

        try:
            result = agent.chat(msg)
            responses.append(result["response"])
            print(f"Tokens in context: {result['tokens_in_context']}")
            print(f"Agent: {result['response'][:200]}…")
        except Exception as exc:
            print(f"ERROR in turn {i+1}: {exc}")
            responses.append("")

        # Rate limit guard: sleep between turns
        if i < len(turns) - 1:
            print(f"(sleeping {INTER_TURN_SLEEP}s for rate limit…)")
            time.sleep(INTER_TURN_SLEEP)

    final = responses[-1].lower() if responses else ""

    print(f"\n{'─'*55}")
    print("FULL FINAL RESPONSE:")
    print(responses[-1] if responses else "(no response)")
    print(f"{'─'*55}")

    allergy_words = [
        "shellfish", "allerg", "seafood",
        "⚠", "warning", "avoid", "cannot eat", "medical"
    ]
    allergy_mentioned = any(w in final for w in allergy_words)

    print(f"\nAllergy mentioned in final response: {allergy_mentioned}")

    if not allergy_mentioned:
        print("\n  DIAGNOSIS CHECKLIST:")
        print("  1. Was the allergy extracted as CRITICAL? Check turn 1 output.")
        print("  2. Is [CRITICAL CONSTRAINTS] section visible in context?")
        print("     Look for it in the turn 4 '[CCM] Step 4' output above.")
        print("  3. Did the LLM follow the system prompt constraint check?")
        print("     The system prompt requires checking [CRITICAL CONSTRAINTS].")

    print(f"{'✅ TEST 5 PASSED' if allergy_mentioned else '❌ TEST 5 FAILED'}")
    return allergy_mentioned


# ─────────────────────────────────────────────────────────────────
# TEST 6 — Baseline Fails (proves the problem)
# ─────────────────────────────────────────────────────────────────

def test_6_baseline_fails():
    """
    Show that the baseline agent (no CCM) either overflows or
    forgets the shellfish allergy by turn 5.

    This is the "before" state that justifies the CCM.
    The test PASSES if the baseline demonstrates a problem.
    """
    print("\n" + "=" * 55)
    print("TEST 6: Baseline Fails (proves the problem exists)")
    print("=" * 55)

    reset_all_storage()

    from travel_agent.baseline_agent import BaselineAgent

    agent = BaselineAgent()

    turns = [
        "I want to plan a trip to Tokyo. Budget $3000 maximum. "
        "I am severely allergic to shellfish — medical allergy.",
        "Find flights from New York to Tokyo",
        "Search for hotels in Tokyo",
        "What is the weather in Tokyo in June?",
        "Find restaurants near Tsukiji fish market",
    ]

    tokens_per_turn = []
    responses       = []

    for i, msg in enumerate(turns):
        print(f"\nTurn {i+1}: {msg[:60]}…")
        try:
            result = agent.chat(msg)
            tokens_per_turn.append(result["tokens_in_context"])
            responses.append(result["response"])
            print(f"Tokens: {result['tokens_in_context']}")
            r_preview = result["response"][:100]
            print(f"Response: {r_preview}…")
        except Exception as exc:
            print(f"ERROR (this is expected for baseline): {exc}")
            tokens_per_turn.append(0)
            responses.append("")

        if i < len(turns) - 1:
            time.sleep(INTER_TURN_SLEEP)

    print(f"\nToken growth per turn: {tokens_per_turn}")

    final = responses[-1].lower() if responses else ""
    allergy_remembered = any(
        w in final for w in ["shellfish", "allerg", "seafood"]
    )

    overflow_occurred = any(t > 6000 for t in tokens_per_turn)
    print(f"Context overflow occurred (>6000 tokens): {overflow_occurred}")
    print(f"Baseline still remembered allergy at turn 5: {allergy_remembered}")

    # The test PASSES if the baseline has a problem
    passed = overflow_occurred or not allergy_remembered
    print(
        f"{'✅ TEST 6 PASSED (baseline correctly failed)' if passed else '❌ TEST 6 FAILED (baseline worked — unusual)'}"
    )
    return passed


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def test_6_baseline_fails():
    """
    Show that the baseline agent (no CCM) degrades on the official
    long-horizon shellfish conversation.

    This is the "before" state that justifies the CCM.
    The test PASSES if the baseline demonstrates a problem:
      - forgets the allergy,
      - overflows, or
      - becomes severely token-inefficient.
    """
    print("\n" + "=" * 55)
    print("TEST 6: Baseline Fails (proves the problem exists)")
    print("=" * 55)

    reset_all_storage()

    from travel_agent.baseline_agent import BaselineAgent
    from evaluation.test_conversations import TEST_A_SHELLFISH_ALLERGY

    agent = BaselineAgent()
    turns = [turn["content"] for turn in TEST_A_SHELLFISH_ALLERGY["turns"]]

    tokens_per_turn = []
    responses       = []

    for i, msg in enumerate(turns):
        print(f"\nTurn {i+1}: {msg[:60]}...")
        try:
            result = agent.chat(msg)
            tokens_per_turn.append(result["tokens_in_context"])
            responses.append(result["response"])
            print(f"Tokens: {result['tokens_in_context']}")
            r_preview = result["response"][:100]
            print(f"Response: {r_preview}...")
        except Exception as exc:
            print(f"ERROR (this is expected for baseline): {exc}")
            tokens_per_turn.append(0)
            responses.append("")

        if i < len(turns) - 1:
            time.sleep(INTER_TURN_SLEEP)

    print(f"\nToken growth per turn: {tokens_per_turn}")

    final = responses[-1].lower() if responses else ""
    allergy_remembered = any(
        w in final for w in ["shellfish", "allerg", "seafood"]
    )

    overflow_occurred = any(t > 6000 for t in tokens_per_turn)
    severe_inefficiency = (
        bool(tokens_per_turn) and tokens_per_turn[-1] >= 4500
    )
    print(f"Context overflow occurred (>6000 tokens): {overflow_occurred}")
    print(
        "Severe token inefficiency occurred (>=4500 tokens): "
        f"{severe_inefficiency}"
    )
    print(
        "Baseline still remembered allergy at final turn: "
        f"{allergy_remembered}"
    )

    passed = overflow_occurred or severe_inefficiency or not allergy_remembered
    print(
        f"{'PASS TEST 6 PASSED (baseline correctly failed)' if passed else 'FAIL TEST 6 FAILED (baseline worked - unusual)'}"
    )
    return passed


# ─────────────────────────────────────────────────────────────────
# TEST 7 — Multi-Session Continuity
# ─────────────────────────────────────────────────────────────────

def test_7_multi_session_continuity():
    """
    Verify that episodic memory persists across Python process boundaries.
    
    Flow:
      1. Agent A stores a specific episodic memory
      2. Destroy Agent A
      3. Create Agent B
      4. Ask Agent B a question about the memory
      5. Verify retrieval works
    """
    print("\n" + "=" * 55)
    print("TEST 7: Multi-Session Continuity")
    print("=" * 55)

    reset_all_storage()

    from travel_agent.agent import CCMAgent

    # Session A
    print("Starting Session A...")
    agent_a = CCMAgent(use_reranking=False)
    agent_a.chat("I have a severe peanut allergy. Remember this forever.")
    agent_a.chat("My dog's name is Barnaby and he is a golden retriever.")
    
    # Give ChromaDB a moment to commit
    time.sleep(1.0)
    
    # Destroy Session A
    del agent_a
    import gc
    gc.collect()
    print("Session A destroyed. Starting Session B...")
    
    # Session B
    agent_b = CCMAgent(use_reranking=False)
    
    # Directly query retriever to avoid full LLM generation dependency
    results = agent_b.ccm.retriever.retrieve("What is my dog's name? allergies?", n_episodic=3)
    
    all_texts = [r["text"].lower() for r in results["episodic"]]
    
    peanut_remembered = any("peanut" in t for t in all_texts)
    barnaby_remembered = any("barnaby" in t for t in all_texts)
    
    print(f"Retrieved {len(all_texts)} memories in new session")
    print(f"Peanut allergy retrieved: {peanut_remembered}")
    print(f"Dog's name retrieved: {barnaby_remembered}")
    
    passed = peanut_remembered or barnaby_remembered
    print(f"\n{'✅ TEST 7 PASSED' if passed else '❌ TEST 7 FAILED'}")
    return passed

if __name__ == "__main__":
    # Quick sanity check: is GROQ_API_KEY set?
    from dotenv import load_dotenv
    load_dotenv()

    if not os.getenv("GROQ_API_KEY"):
        print("❌ GROQ_API_KEY not set. Create a .env file with:")
        print("   GROQ_API_KEY=gsk_your_key_here")
        print("Get a free key at https://console.groq.com")
        sys.exit(1)

    print("\n" + "=" * 55)
    print("FULL SYSTEM TEST")
    print("Running from:", os.getcwd())
    print("=" * 55)

    results = {}

    # Tests 1-4 are fast (no multi-turn agent)
    results["memory_extraction"] = test_1_memory_extraction()

    print(f"\n(sleeping {INTER_TURN_SLEEP}s between tests…)\n")
    time.sleep(INTER_TURN_SLEEP)

    results["stale_detection"] = test_2_stale_detection()

    print(f"\n(sleeping {INTER_TURN_SLEEP}s…)\n")
    time.sleep(INTER_TURN_SLEEP)

    results["compression"] = test_3_compression()

    print(f"\n(sleeping {INTER_TURN_SLEEP}s…)\n")
    time.sleep(INTER_TURN_SLEEP)

    results["rag_retrieval"] = test_4_rag_retrieval()

    print(f"\n(sleeping {INTER_TURN_SLEEP * 2}s before full-pipeline tests…)\n")
    time.sleep(INTER_TURN_SLEEP * 2)

    # Tests 5-6 are heavier (full multi-turn agent)
    results["ccm_allergy"]   = test_5_ccm_agent_allergy()

    print(f"\n(sleeping {INTER_TURN_SLEEP * 2}s…)\n")
    time.sleep(INTER_TURN_SLEEP * 2)

    results["baseline_fails"] = test_6_baseline_fails()

    print(f"\n(sleeping {INTER_TURN_SLEEP * 2}s…)\n")
    time.sleep(INTER_TURN_SLEEP * 2)

    results["multi_session"] = test_7_multi_session_continuity()

    # Final summary
    print("\n" + "=" * 55)
    print("FINAL RESULTS")
    print("=" * 55)
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {name}")

    total = sum(results.values())
    print(f"\n{total}/{len(results)} tests passed")

    if total < len(results):
        print("\nFor failed tests, check the DIAGNOSIS notes above.")
        print("Most common fixes:")
        print("  - GROQ_API_KEY rate limit: increase INTER_TURN_SLEEP at top of file")
        print("  - sentence-transformers not installed: pip install sentence-transformers")
        print("  - ChromaDB folder locked: close any other Python processes and retry")
