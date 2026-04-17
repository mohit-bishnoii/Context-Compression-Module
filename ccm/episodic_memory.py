# ccm/episodic_memory.py
#
# Tier 2: Episodic Memory
#
# WHAT IT STORES:
#   Summaries of conversation chunks.
#   Created every N turns by summarizing what happened.
#   Example: "Searched flights to Tokyo. ANA $780 direct approved.
#              Hilton hotel rejected at $220, over budget."
#
# HOW IT WORKS:
#   Each summary is embedded into a vector and stored in ChromaDB.
#   When a new user message arrives, we embed the query and find
#   the most semantically similar summaries.
#   Only the top K relevant summaries get injected into the prompt.
#
# WHY THIS IS NOT SIMPLE RAG:
#   Simple RAG: search static documents, paste results
#   Our system: dynamically CREATES memories from conversation,
#               compresses before storing,
#               marks stale when overridden,
#               scores retrieved results by relevance,
#               respects token budget during injection
#
# AGENT-CENTRIC:
#   This class knows nothing about travel.
#   It stores whatever summaries the CCM creates.
#   Works identically for any agent domain.

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
import uuid
from datetime import datetime
from typing import Optional
import chromadb
from sentence_transformers import SentenceTransformer

# ── Configuration ──────────────────────────────────────────────
CHROMA_PATH = "./data/chroma_db"
COLLECTION_NAME = "episodic_memory"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5          # How many results to retrieve by default
SIMILARITY_THRESHOLD = 0.3  # Minimum similarity score to include
                            # Range: 0.0 (anything) to 1.0 (exact match)
                            # 0.3 means loosely related = included
                            # 0.7 means only very similar = strict

# Shared embedding model — loaded once, reused everywhere
# Loading takes ~2 seconds, so we do it at module level
_embedding_model = None


def get_embedding_model() -> SentenceTransformer:
    """
    Load embedding model once and cache it.
    Subsequent calls return the cached instance instantly.
    """
    global _embedding_model
    if _embedding_model is None:
        print("[Embeddings] Loading all-MiniLM-L6-v2 model...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print("[Embeddings] Model loaded")
    return _embedding_model


def embed(text: str) -> list:
    """
    Convert text to a vector (list of 384 floats).
    This is what enables semantic similarity search.
    
    Similar meaning → similar vector → found by RAG
    Different meaning → different vector → not retrieved
    """
    model = get_embedding_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


class EpisodicMemory:
    """
    Tier 2: Episodic Memory
    
    Stores summaries of conversation episodes in ChromaDB.
    Each summary is embedded as a vector for semantic search.
    
    Lifecycle of an episodic memory:
      1. Every N turns, CCM calls create_episode(turns)
      2. The turns are summarized by the LLM (via compressor)
      3. The summary is embedded and stored in ChromaDB
      4. On next query, retrieve() finds relevant episodes
      5. If something is cancelled, mark_stale_by_content() hides it
    
    The key insight: we store SUMMARIES not raw turns.
    A 10-turn chunk that uses 2000 tokens becomes a
    60-word summary using ~80 tokens. That is 25x compression
    before we even do retrieval.
    """

    def __init__(self):
        os.makedirs(CHROMA_PATH, exist_ok=True)
        os.makedirs("data", exist_ok=True)

        # Connect to ChromaDB (persistent — survives restarts)
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)

        # Get or create the episodic collection
        # ChromaDB collections are like database tables
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "description": "Episodic conversation summaries",
                "hnsw:space": "cosine"  # Use cosine similarity
                                         # Best for text embeddings
            }
        )
        print(
            f"[EpisodicMemory] Ready. "
            f"Contains {self.collection.count()} entries."
        )

    def add(
        self,
        summary_text: str,
        turn_range: tuple = (0, 0),
        metadata: Optional[dict] = None
    ) -> str:
        """
        Store a new episodic summary.
        
        Parameters:
          summary_text: The compressed summary text
          turn_range:   (start_turn, end_turn) this summary covers
          metadata:     Optional extra metadata to store
        
        Returns:
          The unique ID assigned to this memory
        
        Example:
          memory_id = episodic.add(
            "Searched Tokyo hotels. Hilton $220 rejected.
             Shinjuku Park $120 shortlisted.",
            turn_range=(3, 7)
          )
        """
        if not summary_text or not summary_text.strip():
            print("[EpisodicMemory] Empty summary, not storing")
            return ""

        # Generate unique ID for this memory
        memory_id = f"ep_{uuid.uuid4().hex[:12]}"

        # Build metadata dict
        # ChromaDB stores this alongside the vector
        # We can filter by metadata during retrieval
        entry_metadata = {
            "turn_start": turn_range[0],
            "turn_end": turn_range[1],
            "created_at": datetime.now().isoformat(),
            "stale": False,   # True = do not retrieve this
            "type": "episodic_summary"
        }

        # Merge any extra metadata passed in
        if metadata:
            entry_metadata.update(metadata)

        # Embed the summary text
        # This converts text → 384 numbers representing meaning
        vector = embed(summary_text)

        # Store in ChromaDB
        # ChromaDB stores: text, vector, metadata, all linked by ID
        self.collection.add(
            documents=[summary_text],
            embeddings=[vector],
            metadatas=[entry_metadata],
            ids=[memory_id]
        )

        print(
            f"[EpisodicMemory] Stored: {memory_id} "
            f"(turns {turn_range[0]}-{turn_range[1]})"
        )
        print(f"  Summary: {summary_text[:80]}...")

        return memory_id

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        exclude_stale: bool = True
        ) -> list:
        """
        Retrieve relevant episodic memories.

        KEY CHANGE: Filter stale entries in Python, not ChromaDB.
        ChromaDB boolean filtering is unreliable across versions.
        We fetch more results and filter manually.
        """
        if self.collection.count() == 0:
            return []

        # Fetch more than needed so we have room to filter stale
        fetch_k = min(top_k * 3, self.collection.count())
        query_vector = embed(query)

        try:
            # Fetch WITHOUT any where filter
            # We will filter stale manually in Python
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=fetch_k,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            print(f"[EpisodicMemory] Query error: {e}")
            return []

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        retrieved = []
        for doc, meta, dist, rid in zip(
            documents, metadatas, distances, ids
        ):
            # MANUAL stale filter in Python — reliable
            if exclude_stale:
                is_stale = meta.get("stale", False)
                # Handle string "true"/"false" from ChromaDB
                if isinstance(is_stale, str):
                    is_stale = is_stale.lower() == "true"
                if is_stale:
                    print(f"[EpisodicMemory] Skipping stale: {doc[:40]}")
                    continue

            similarity = 1.0 - (dist / 2.0)

            if similarity < SIMILARITY_THRESHOLD:
                continue

            retrieved.append({
                "id": rid,
                "text": doc,
                "similarity": round(similarity, 3),
                "metadata": meta
            })

            # Stop once we have enough
            if len(retrieved) >= top_k:
                break

        if retrieved:
            print(
                f"[EpisodicMemory] Retrieved {len(retrieved)} memories"
            )
            for r in retrieved:
                print(f"  [{r['similarity']:.2f}] {r['text'][:60]}")
        else:
            print("[EpisodicMemory] No relevant memories found")

        return retrieved

    def mark_stale_by_content(self, substring: str) -> int:
        """
        Mark all entries containing ANY word from substring as stale.
        
        KEY CHANGE: Instead of matching the full substring,
        we extract significant words and match any of them.
        This handles cases where the cancelled value phrase
        differs from the stored memory text.
        
        Example:
        cancelled value: "Bali beach vacation"
        stored memory:   "Researched Bali resorts..."
        "Bali" is the significant word → matches correctly
        """
        if self.collection.count() == 0:
            return 0

        # Extract significant words (ignore common words)
        stop_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were',
            'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'trip',
            'plan', 'planned', 'vacation', 'holiday'
        }
        
        words = substring.lower().split()
        significant_words = [
            w.strip('.,!?') for w in words
            if w.strip('.,!?') not in stop_words
            and len(w.strip('.,!?')) > 3
        ]
        
        if not significant_words:
            # Fall back to full substring if no significant words
            significant_words = [substring.lower()]
        
        print(
            f"[EpisodicMemory] Marking stale entries containing: "
            f"{significant_words}"
        )

        try:
            all_entries = self.collection.get(
                include=["documents", "metadatas"]
            )
        except Exception as e:
            print(f"[EpisodicMemory] Error getting entries: {e}")
            return 0

        documents = all_entries.get("documents", [])
        metadatas = all_entries.get("metadatas", [])
        ids = all_entries.get("ids", [])

        stale_count = 0

        for doc, meta, entry_id in zip(documents, metadatas, ids):
            doc_lower = doc.lower()
            
            # Check if ANY significant word appears in the document
            if any(word in doc_lower for word in significant_words):
                if meta.get("stale", False):
                    continue

                updated_meta = dict(meta)
                updated_meta["stale"] = True
                updated_meta["staled_at"] = datetime.now().isoformat()
                updated_meta["stale_reason"] = f"Cancelled: {substring}"

                self.collection.update(
                    ids=[entry_id],
                    metadatas=[updated_meta]
                )
                stale_count += 1
                print(
                    f"[EpisodicMemory] Marked stale: "
                    f"{doc[:60]}..."
                )

        if stale_count > 0:
            print(
                f"[EpisodicMemory] Total marked stale: "
                f"{stale_count} entries"
            )

        return stale_count

    def get_all_active(self) -> list:
        """
        Get all non-stale entries. Used for debugging and UI display.
        """
        if self.collection.count() == 0:
            return []

        try:
            results = self.collection.get(
                where={"stale": {"$eq": False}},
                include=["documents", "metadatas"]
            )
            entries = []
            for doc, meta, entry_id in zip(
                results["documents"],
                results["metadatas"],
                results["ids"]
            ):
                entries.append({
                    "id": entry_id,
                    "text": doc,
                    "metadata": meta
                })
            return entries
        except Exception as e:
            print(f"[EpisodicMemory] Error getting all: {e}")
            return []

    def get_count(self) -> dict:
        """Return count of total and stale entries."""
        total = self.collection.count()
        if total == 0:
            return {"total": 0, "active": 0, "stale": 0}

        try:
            stale_results = self.collection.get(
                where={"stale": {"$eq": True}}
            )
            stale_count = len(stale_results.get("ids", []))
            return {
                "total": total,
                "active": total - stale_count,
                "stale": stale_count
            }
        except Exception:
            return {"total": total, "active": total, "stale": 0}

    def reset(self):
        """
        Delete all episodic memories. Called on new conversation.
        """
        try:
            self.client.delete_collection(COLLECTION_NAME)
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            print("[EpisodicMemory] Reset complete")
        except Exception as e:
            print(f"[EpisodicMemory] Reset error: {e}")