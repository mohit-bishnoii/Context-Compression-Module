# ccm/episodic_memory.py
#
# Tier 2: Episodic Memory
#
# Stores compressed summaries of conversation chunks.
# Every N turns, CCMCore summarises the last N turns into
# one short paragraph and calls episodic.add().
# Those summaries are embedded as vectors and stored in ChromaDB.
# On the next user turn, retrieve() finds the most relevant
# summaries by cosine similarity.
#
# STALE RULE (critical):
#   All "stale" metadata values are stored as the STRING "false" or
#   "true" — NEVER Python booleans.
#   ChromaDB 0.5.x boolean filtering via `where` clauses is
#   unreliable (known upstream issue). We filter stale entries
#   manually in Python after fetching.
#
# AGENT-CENTRIC: knows nothing about travel, allergies, or Tokyo.
# Stores whatever text the CCMCore gives it.

import os
import uuid
from datetime import datetime
from typing import Optional

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"]     = "False"

import chromadb
from sentence_transformers import SentenceTransformer

# ── Config ──────────────────────────────────────────────────────
CHROMA_PATH       = "./data/chroma_db"
COLLECTION_NAME   = "episodic_memory"
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"
DEFAULT_TOP_K     = 5
SIMILARITY_THRESHOLD = 0.25   # lowered slightly so distant-but-relevant
                               # memories still get retrieved

# Shared embedding model — loaded once, cached forever
_embedding_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Load the embedding model once, then reuse the cached instance."""
    global _embedding_model
    if _embedding_model is None:
        print("[Embeddings] Loading all-MiniLM-L6-v2 …")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print("[Embeddings] Model ready")
    return _embedding_model


def embed(text: str) -> list:
    """Convert a text string → 384-float vector for semantic search."""
    model = get_embedding_model()
    return model.encode(text, normalize_embeddings=True).tolist()


# ── Stop-words for stale matching ───────────────────────────────
_STOP = {
    "a","an","the","is","are","was","were","and","or","but",
    "in","on","at","to","for","of","with","by","from",
    "trip","plan","planned","vacation","holiday","let","do",
    "us","me","my","its","this","that","have","has","will",
    "would","could","should","i","we","you","it","be","been",
}


class EpisodicMemory:
    """
    Tier 2 — Episodic Memory.

    Public API (called by CCMCore):
      add(summary_text, turn_range)            → memory_id: str
      retrieve(query, top_k)                   → list[dict]
      mark_stale_by_content(substring)         → int (count marked)
      get_all_active()                         → list[dict]
      get_count()                              → dict
      reset()                                  → None

    Internal storage:
      ChromaDB persistent collection named "episodic_memory".
      Every document has metadata including stale="false"|"true".
    """

    def __init__(self):
        os.makedirs(CHROMA_PATH, exist_ok=True)
        os.makedirs("data", exist_ok=True)

        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print(
            f"[EpisodicMemory] Ready — {self.collection.count()} entries."
        )

    # ── Write ──────────────────────────────────────────────────

    def add(
        self,
        summary_text: str,
        turn_range: tuple = (0, 0),
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Store a new episode summary.

        Parameters
        ----------
        summary_text : str
            The compressed paragraph produced by _create_episode_summary().
        turn_range : (int, int)
            The conversation turn span this summary covers.
        metadata : dict | None
            Optional extra metadata (e.g., {"topic": "flights"}).

        Returns
        -------
        str
            Unique memory ID, e.g. "ep_3f8a12b0c4d1".
            Returns "" if nothing was stored.
        """
        if not summary_text or not summary_text.strip():
            print("[EpisodicMemory] Empty summary — skipping")
            return ""

        memory_id = f"ep_{uuid.uuid4().hex[:12]}"

        entry_meta = {
            "turn_start": turn_range[0],
            "turn_end":   turn_range[1],
            "created_at": datetime.now().isoformat(),
            "stale":      "false",   # ← always a STRING
            "type":       "episodic_summary",
        }
        if metadata:
            # Handle topic and other custom metadata
            # Coerce any booleans to strings to be safe
            for k, v in metadata.items():
                if k == "topic":
                    # topic can be stored as-is (string)
                    entry_meta[k] = str(v)
                else:
                    entry_meta[k] = ("true" if v else "false") if isinstance(v, bool) else v

        vector = embed(summary_text)

        self.collection.add(
            documents=[summary_text],
            embeddings=[vector],
            metadatas=[entry_meta],
            ids=[memory_id],
        )
        print(
            f"[EpisodicMemory] Stored {memory_id} "
            f"(turns {turn_range[0]}–{turn_range[1]}): "
            f"{summary_text[:80]}…"
        )
        return memory_id

    # ── Read ───────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        exclude_stale: bool = True,
    ) -> list:
        """
        Return the most semantically relevant episodic summaries.

        How it works
        ------------
        1. Embed the query into a vector.
        2. Ask ChromaDB for the nearest ``top_k * 3`` entries
           (we fetch more so we have room to filter).
        3. Filter stale entries in Python (not via ChromaDB where clause).
        4. Filter by similarity threshold.
        5. Return at most ``top_k`` results.

        Returns
        -------
        list[dict]
            Each dict: {id, text, similarity, metadata}
        """
        total = self.collection.count()
        if total == 0:
            return []

        fetch_k = min(top_k * 3, total)
        query_vector = embed(query)

        try:
            raw = self.collection.query(
                query_embeddings=[query_vector],
                n_results=fetch_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            print(f"[EpisodicMemory] query() error: {exc}")
            return []

        docs      = raw.get("documents", [[]])[0]
        metas     = raw.get("metadatas",  [[]])[0]
        distances = raw.get("distances",  [[]])[0]
        ids       = raw.get("ids",        [[]])[0]

        results = []
        for doc, meta, dist, rid in zip(docs, metas, distances, ids):
            # ── Stale filter (Python-side) ───────────────────
            if exclude_stale:
                raw_stale = meta.get("stale", "false")
                is_stale = (
                    raw_stale if isinstance(raw_stale, bool)
                    else str(raw_stale).lower() == "true"
                )
                if is_stale:
                    print(f"[EpisodicMemory] Skipping stale: {doc[:50]}")
                    continue

            # ── Similarity filter ───────────────────────────
            # ChromaDB cosine distance: 0=identical, 2=opposite
            # Similarity = 1 − (distance / 2)
            similarity = 1.0 - (dist / 2.0)
            if similarity < SIMILARITY_THRESHOLD:
                continue

            results.append({
                "id":         rid,
                "text":       doc,
                "similarity": round(similarity, 3),
                "metadata":   meta,
            })

            if len(results) >= top_k:
                break

        if results:
            print(f"[EpisodicMemory] Retrieved {len(results)} memories:")
            for r in results:
                print(f"  [{r['similarity']:.2f}] {r['text'][:60]}")
        else:
            print("[EpisodicMemory] No relevant memories found")

        return results

    # ── Stale management ────────────────────────────────────────

    def mark_stale_by_content(self, cancelled_value: str) -> int:
        """
        Mark all entries that contain any significant word from
        ``cancelled_value`` as stale so they are never retrieved again.

        Example
        -------
        cancelled_value = "Bali beach vacation"
        significant words → ["bali", "beach"]
        Entry "Researched Bali resorts…" → contains "bali" → marked stale ✓

        Returns
        -------
        int : number of entries marked stale
        """
        total = self.collection.count()
        if total == 0:
            return 0

        # Extract significant words
        words = cancelled_value.lower().split()
        significant = [
            w.strip(".,!?'\"-") for w in words
            if w.strip(".,!?'\"-") not in _STOP
            and len(w.strip(".,!?'\"-")) > 2
        ]
        if not significant:
            significant = [cancelled_value.lower()[:20]]

        print(
            f"[EpisodicMemory] Marking stale entries "
            f"containing any of: {significant}"
        )

        try:
            all_entries = self.collection.get(
                include=["documents", "metadatas"]
            )
        except Exception as exc:
            print(f"[EpisodicMemory] get() error: {exc}")
            return 0

        docs  = all_entries.get("documents", [])
        metas = all_entries.get("metadatas",  [])
        ids   = all_entries.get("ids",        [])

        stale_count = 0
        for doc, meta, eid in zip(docs, metas, ids):
            doc_lower = doc.lower()

            # Check if already stale
            raw_stale = meta.get("stale", "false")
            already_stale = (
                raw_stale if isinstance(raw_stale, bool)
                else str(raw_stale).lower() == "true"
            )
            if already_stale:
                continue

            # Check if this entry mentions the cancelled topic
            if not any(word in doc_lower for word in significant):
                continue

            # Mark it stale
            updated = dict(meta)
            updated["stale"]        = "true"   # ← STRING always
            updated["staled_at"]    = datetime.now().isoformat()
            updated["stale_reason"] = f"Cancelled: {cancelled_value}"

            try:
                self.collection.update(ids=[eid], metadatas=[updated])
                stale_count += 1
                print(f"[EpisodicMemory] Marked stale: {doc[:60]}")
            except Exception as exc:
                print(f"[EpisodicMemory] update() error on {eid}: {exc}")

        print(f"[EpisodicMemory] Total marked stale: {stale_count}")
        return stale_count

    # ── Utility ────────────────────────────────────────────────

    def get_all_active(self) -> list:
        """Return all non-stale entries. Filters in Python."""
        if self.collection.count() == 0:
            return []
        try:
            raw   = self.collection.get(include=["documents", "metadatas"])
        except Exception as exc:
            print(f"[EpisodicMemory] get_all_active error: {exc}")
            return []

        entries = []
        for doc, meta, eid in zip(
            raw.get("documents", []),
            raw.get("metadatas",  []),
            raw.get("ids",        []),
        ):
            raw_stale = meta.get("stale", "false")
            is_stale  = (
                raw_stale if isinstance(raw_stale, bool)
                else str(raw_stale).lower() == "true"
            )
            if not is_stale:
                entries.append({"id": eid, "text": doc, "metadata": meta})
        return entries

    def get_count(self) -> dict:
        """Return {total, active, stale}. Filters in Python."""
        total = self.collection.count()
        if total == 0:
            return {"total": 0, "active": 0, "stale": 0}

        try:
            raw      = self.collection.get(include=["metadatas"])
            all_meta = raw.get("metadatas", [])
        except Exception:
            return {"total": total, "active": total, "stale": 0}

        stale_count = sum(
            1 for m in all_meta
            if (
                m.get("stale") if isinstance(m.get("stale"), bool)
                else str(m.get("stale", "false")).lower() == "true"
            )
        )
        return {"total": total, "active": total - stale_count, "stale": stale_count}

    def reset(self):
        """Delete the entire collection and recreate it fresh."""
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print("[EpisodicMemory] Reset complete")