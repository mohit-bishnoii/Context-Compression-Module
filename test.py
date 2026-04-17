import sys
import os
import time
import json
import gc

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

sys.path.append('.')

# Global clients to track and close
_chroma_clients = []

def register_chroma_client(client):
    """Track ChromaDB clients so we can close them."""
    _chroma_clients.append(client)

def reset_all_storage():
    """
    Close all ChromaDB connections first, then delete files.
    Windows requires files to be released before deletion.
    """
    import shutil
    import chromadb

    # Step 1: Close all tracked ChromaDB clients
    global _chroma_clients
    for client in _chroma_clients:
        try:
            # Reset client to release file handles
            client.reset()
        except Exception:
            pass
    _chroma_clients.clear()

    # Step 2: Force garbage collection
    # This releases any lingering Python references
    gc.collect()
    time.sleep(1)  # Give Windows time to release locks

    # Step 3: Reset working memory
    os.makedirs("data", exist_ok=True)
    with open("data/working_memory.json", "w") as f:
        json.dump({
            "facts": {
                "critical": [],
                "important": [],
                "contextual": []
            },
            "decisions": [],
            "cancelled": [],
            "turn_count": 0,
            "conversation_id": "",
            "last_updated": ""
        }, f)

    # Step 4: Try to delete ChromaDB folder
    chroma_path = "./data/chroma_db"
    if os.path.exists(chroma_path):
        try:
            shutil.rmtree(chroma_path)
            print("[Reset] ChromaDB folder deleted")
        except PermissionError:
            # If still locked, just clear collections instead
            print("[Reset] Cannot delete folder (locked), clearing collections...")
            try:
                client = chromadb.PersistentClient(path=chroma_path)
                # Delete each collection individually
                for collection in client.list_collections():
                    try:
                        client.delete_collection(collection.name)
                        print(f"[Reset] Deleted collection: {collection.name}")
                    except Exception as e:
                        print(f"[Reset] Could not delete {collection.name}: {e}")
                del client
                gc.collect()
            except Exception as e:
                print(f"[Reset] Collection clear failed: {e}")

    os.makedirs(chroma_path, exist_ok=True)
    time.sleep(0.5)
    print("[Reset] All storage cleared")

def test_1_memory_extraction():
    """Test fact extraction with correct priority."""
    print("\n" + "="*55)
    print("TEST 1: Memory Extraction Priority")
    print("="*55)

    reset_all_storage()

    from ccm.memory_store import WorkingMemory
    from ccm.extractor import MemoryExtractor

    memory = WorkingMemory()
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
        print(f"  [{f['priority']:11}] {f['key']}: {f['value']}")

    print("\nFormatted memory:")
    print(memory.format_for_prompt())

    critical = memory.get_critical_facts()
    critical_values = [f['value'].lower() for f in critical]
    allergy_critical = any('shellfish' in v for v in critical_values)

    print(f"\nShellfish allergy is CRITICAL: {allergy_critical}")
    result = allergy_critical
    print(f"{'✅ TEST 1 PASSED' if result else '❌ TEST 1 FAILED'}")
    return result


def test_2_stale_detection():
    """Test that cancelled context is hidden from retrieval."""
    print("\n" + "="*55)
    print("TEST 2: Stale Context Detection")
    print("="*55)

    reset_all_storage()

    from ccm.memory_store import WorkingMemory
    from ccm.stale_detector import StaleDetector
    from ccm.episodic_memory import EpisodicMemory
    from ccm.semantic_memory import SemanticMemory

    memory = WorkingMemory()
    episodic = EpisodicMemory()
    semantic = SemanticMemory()
    detector = StaleDetector()

    # Add Bali to memory
    memory.add_facts([{
        "key": "destination_primary",
        "value": "Bali beach vacation",
        "category": "decision",
        "priority": "important"
    }])

    # Add Bali to episodic
    ep_id = episodic.add(
        "Researched Bali resorts. Found Seminyak Beach Hotel $180/night.",
        turn_range=(1, 3)
    )
    print(f"Stored episodic entry: {ep_id}")

    # Verify it is retrievable BEFORE stale marking
    before_results = episodic.retrieve("Bali resorts")
    print(f"Before pivot - Bali in episodic: {len(before_results)}")
    assert len(before_results) > 0, "Should find Bali before pivot"

    # User cancels Bali
    pivot_msg = "Scratch Bali, let us do Switzerland instead."
    result = detector.check_and_clean(
        pivot_msg, memory, episodic, semantic
    )

    print(f"Override detected: {result['has_override']}")
    time.sleep(1)  # Wait for ChromaDB to update

    # Verify Bali is GONE after stale marking
    after_results = episodic.retrieve("Bali resorts")
    print(f"After pivot - Bali in episodic: {len(after_results)}")

    # Check memory has cancelled Bali
    cancelled = memory.get_all().get("cancelled", [])
    print(f"Cancelled in memory: {cancelled}")

    passed = (
        result['has_override'] and
        len(after_results) == 0
    )
    print(f"{'✅ TEST 2 PASSED' if passed else '❌ TEST 2 FAILED'}")
    return passed


def test_3_compression():
    """Test tool result compression ratio and constraint flagging."""
    print("\n" + "="*55)
    print("TEST 3: Tool Result Compression")
    print("="*55)

    reset_all_storage()

    from ccm.compressor import ToolCompressor
    from travel_agent.tools import places_search

    compressor = ToolCompressor()

    raw = places_search("Tsukiji Tokyo", "restaurants")
    raw_str = str(raw)
    raw_tokens = len(raw_str) // 4

    constraints = ["severely allergic to shellfish"]
    compressed = compressor.compress(raw, "places_search", constraints)
    compressed_tokens = len(compressed) // 4
    ratio = raw_tokens / max(compressed_tokens, 1)

    print(f"Raw tokens:        {raw_tokens}")
    print(f"Compressed tokens: {compressed_tokens}")
    print(f"Ratio:             {ratio:.1f}x")
    print(f"\nCompressed:\n{compressed}")

    shellfish_flagged = any(w in compressed.lower() for w in [
        'shellfish', '⚠️', 'allergy', 'avoid', 'warning'
    ])
    print(f"\nShellfish flagged: {shellfish_flagged}")

    passed = ratio > 2.0 and shellfish_flagged
    print(f"{'✅ TEST 3 PASSED' if passed else '❌ TEST 3 FAILED'}")
    return passed


def test_4_rag_retrieval():
    """Test RAG finds relevant memories and ignores irrelevant ones."""
    print("\n" + "="*55)
    print("TEST 4: RAG Retrieval")
    print("="*55)

    reset_all_storage()

    from ccm.episodic_memory import EpisodicMemory
    from ccm.semantic_memory import SemanticMemory
    from ccm.retriever import Retriever

    ep = EpisodicMemory()
    sem = SemanticMemory()

    ep.add(
        "User severely allergic to shellfish. Budget $3000.",
        turn_range=(0, 1)
    )
    ep.add(
        "Booked ANA flight NYC to Tokyo $780 direct.",
        turn_range=(2, 3)
    )
    sem.add(
        "Tsukiji restaurants: Sushi Dai shellfish heavy avoid. "
        "Odayasu safe traditional Japanese.",
        tool_name="places_search",
        query_used="restaurants near Tsukiji",
        turn_number=4
    )

    retriever = Retriever(ep, sem, use_reranking=False)

    print("\nQuery: 'dinner restaurants Tsukiji shellfish'")
    results = retriever.retrieve(
        "dinner restaurants Tsukiji shellfish",
        n_episodic=3,
        n_semantic=2
    )

    total = len(results['episodic']) + len(results['semantic'])
    print(f"Total retrieved: {total}")

    # The allergy memory should be retrieved
    all_texts = (
        [r['text'] for r in results['episodic']] +
        [r['text'] for r in results['semantic']]
    )
    allergy_retrieved = any(
        'shellfish' in t.lower() or 'allergy' in t.lower()
        for t in all_texts
    )
    print(f"Allergy info retrieved: {allergy_retrieved}")

    passed = total > 0 and allergy_retrieved
    print(f"{'✅ TEST 4 PASSED' if passed else '❌ TEST 4 FAILED'}")
    return passed


def test_5_ccm_agent_allergy():
    """
    KEY TEST: Allergy stated at turn 1, must be flagged at turn 4.
    """
    print("\n" + "="*55)
    print("TEST 5: CCM Agent Remembers Allergy")
    print("="*55)

    reset_all_storage()

    from travel_agent.agent import CCMAgent

    agent = CCMAgent(use_reranking=False)

    turns = [
        (
            "I want to plan a trip to Tokyo. "
            "Budget is $3000 maximum. "
            "I am severely allergic to shellfish. "
            "This is a medical allergy — I cannot eat shellfish."
        ),
        "Find flights from New York to Tokyo in June",
        "Find hotels in Tokyo",
        # KEY: Must flag shellfish allergy in response
        "Find dinner restaurants near Tsukiji fish market in Tokyo"
    ]

    responses = []
    for i, msg in enumerate(turns):
        print(f"\nTurn {i+1}: {msg[:70]}...")
        result = agent.chat(msg)
        responses.append(result['response'])
        print(f"Tokens: {result['tokens_in_context']}")
        if i == len(turns) - 1:
            print(f"\nFINAL RESPONSE:\n{result['response']}")

    # Check final response mentions allergy
    final = responses[-1].lower()
    allergy_words = [
        'shellfish', 'allergy', 'allergic',
        'seafood', '⚠️', 'warning', 'avoid',
        'cannot eat', 'medical'
    ]
    allergy_remembered = any(w in final for w in allergy_words)

    print(f"\nAllergy mentioned: {allergy_remembered}")
    print(f"{'✅ TEST 5 PASSED' if allergy_remembered else '❌ TEST 5 FAILED'}")
    return allergy_remembered


def test_6_baseline_fails():
    """Baseline overflows — proves the problem."""
    print("\n" + "="*55)
    print("TEST 6: Baseline Fails (proves problem)")
    print("="*55)

    reset_all_storage()

    from travel_agent.baseline_agent import BaselineAgent

    agent = BaselineAgent()

    turns = [
        "I want to plan a trip to Tokyo. Budget $3000. "
        "I am severely allergic to shellfish.",
        "Find flights from New York to Tokyo",
        "Search for hotels in Tokyo",
        "What is the weather in Tokyo?",
        "Find restaurants near Tsukiji market"
    ]

    tokens_per_turn = []
    responses = []
    for i, msg in enumerate(turns):
        print(f"\nTurn {i+1}: {msg[:60]}...")
        result = agent.chat(msg)
        tokens_per_turn.append(result['tokens_in_context'])
        responses.append(result['response'])
        print(f"Tokens: {result['tokens_in_context']}")
        if 'ERROR' in result['response'] or 'error' in result['response'].lower():
            print(f"⚠️  Error/overflow detected")

    print(f"\nToken growth: {tokens_per_turn}")

    # Check final response for allergy
    final = responses[-1].lower() if responses else ""
    allergy_remembered = any(
        w in final for w in ['shellfish', 'allergy', 'allergic']
    )

    overflow_occurred = any(t > 6000 for t in tokens_per_turn)
    print(f"Context overflow occurred: {overflow_occurred}")
    print(f"Baseline remembered allergy: {allergy_remembered}")

    # Pass if baseline fails (either overflow or forgot allergy)
    passed = overflow_occurred or not allergy_remembered
    print(f"{'✅ TEST 6 PASSED (baseline correctly failed)' if passed else '❌ TEST 6 FAILED'}")
    return passed


if __name__ == "__main__":
    print("\n" + "="*55)
    print("FULL SYSTEM TEST")
    print("="*55)

    results = {}
    results['memory_extraction'] = test_1_memory_extraction()
    results['stale_detection'] = test_2_stale_detection()
    results['compression'] = test_3_compression()
    results['rag_retrieval'] = test_4_rag_retrieval()
    results['ccm_allergy'] = test_5_ccm_agent_allergy()
    results['baseline_fails'] = test_6_baseline_fails()

    print("\n" + "="*55)
    print("FINAL RESULTS")
    print("="*55)
    for name, passed in results.items():
        print(f"  {'✅ PASS' if passed else '❌ FAIL'}  {name}")

    total = sum(results.values())
    print(f"\n{total}/{len(results)} tests passed")