# ccm/semantic_memory.py
#
# Tier 3: Semantic / Archived Memory
#
# Stores COMPRESSED tool results for later RAG retrieval.
# When the agent calls places_search("hotels Tokyo") and gets
# 600 tokens back, the CCM compresses it to ~80 tokens and
# stores that compressed version here.
#
# Later, when the user asks "remind me what hotels we found",
# retrieve() finds the right archived result by semantic similarity.
#
# KEY DIFFERENCES from EpisodicMemory:
#   - Documents are tool results, not conversation summaries
#   - Metadata includes tool_name and query_used
#   - The TEXT embedded is "Query: X\nResult: Y" so the vector
#     captures both the question and the answer
#
# STALE RULE: identical to EpisodicMemory — string "false"/"true".

import os
import uuid
from datetime import datetime
from typing import Optional

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"]     = "False"

import chromadb
from ccm.episodic_memory import embed   # reuse the same embedding fn

CHROMA_PATH      = "./data/chroma_db"
COLLECTION_NAME  = "semantic_memory"
DEFAULT_TOP_K    = 3
SIMILARITY_THRESHOLD = 0.25

_STOP = {
    "a","an","the","is","are","was","were","and","or","but",
    "in","on","at","to","for","of","with","by","from",
    "trip","plan","planned","vacation","holiday","let","do",
    "us","me","my","its","this","that","have","has","will",
    "would","could","should","i","we","you","it","be","been",
}


class SemanticMemory:
    """
    Tier 3 — Semantic / Archived Memory.

    Public API (called by CCMCore):
      add(compressed_result, tool_name, query_used, turn_number) → id
      retrieve(query, top_k, tool_filter, exclude_stale)         → list
      mark_stale_by_content(substring)                           → int
      get_all_active()                                           → list
      get_count()                                                → dict
      reset()                                                    → None
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
            f"[SemanticMemory] Ready — {self.collection.count()} entries."
        )

    # ── Write ──────────────────────────────────────────────────

    def add(
        self,
        compressed_result: str,
        tool_name: str,
        query_used: str,
        turn_number: int = 0,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Store a compressed tool result.

        The text that gets EMBEDDED is:
          "Query: <query_used>\nResult: <compressed_result>"
        This means future queries about the same topic will find
        this entry even if they use slightly different words.

        Parameters
        ----------
        compressed_result : str
            The ~80-token summary produced by ToolCompressor.
        tool_name : str
            e.g. "places_search", "web_search"
        query_used : str
            The query that was passed to the tool.
        turn_number : int
            Conversation turn (for debugging / UI display).

        Returns
        -------
        str
            Unique memory ID, e.g. "sem_4a1b2c3d4e5f".
        """
        if not compressed_result or not compressed_result.strip():
            print("[SemanticMemory] Empty result — skipping")
            return ""

        memory_id = f"sem_{uuid.uuid4().hex[:12]}"

        # Embed query + result together
        embeddable = f"Query: {query_used}\nResult: {compressed_result}"

        entry_meta = {
            "tool_name":  tool_name,
            "query_used": query_used,
            "turn_number": turn_number,
            "created_at": datetime.now().isoformat(),
            "stale":      "false",   # ← STRING always
            "type":       "tool_result",
        }
        if metadata:
            for k, v in metadata.items():
                entry_meta[k] = ("true" if v else "false") if isinstance(v, bool) else v

        vector = embed(embeddable)

        self.collection.add(
            documents=[compressed_result],
            embeddings=[vector],
            metadatas=[entry_meta],
            ids=[memory_id],
        )
        print(
            f"[SemanticMemory] Stored [{tool_name}] {memory_id}: "
            f"{compressed_result[:200]}…"
        )
        return memory_id

    # ── Read ───────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        tool_filter: Optional[str] = None,
        exclude_stale: bool = True,
    ) -> list:
        """
        Find relevant archived tool results.

        Parameters
        ----------
        query : str
            Current user message or topic.
        top_k : int
            Max results to return.
        tool_filter : str | None
            If set, only return results from this tool.
        exclude_stale : bool
            Skip entries marked stale.

        Returns
        -------
        list[dict]
            Each dict: {id, text, similarity, tool_name, query_used, metadata}
        """
        total = self.collection.count()
        if total == 0:
            return []

        fetch_k      = min(top_k * 3, total)
        query_vector = embed(query)

        try:
            raw = self.collection.query(
                query_embeddings=[query_vector],
                n_results=fetch_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            print(f"[SemanticMemory] query() error: {exc}")
            return []

        docs      = raw.get("documents", [[]])[0]
        metas     = raw.get("metadatas",  [[]])[0]
        distances = raw.get("distances",  [[]])[0]
        ids       = raw.get("ids",        [[]])[0]

        results = []
        for doc, meta, dist, rid in zip(docs, metas, distances, ids):
            # Stale filter
            if exclude_stale:
                raw_stale = meta.get("stale", "false")
                is_stale  = (
                    raw_stale if isinstance(raw_stale, bool)
                    else str(raw_stale).lower() == "true"
                )
                if is_stale:
                    print(f"[SemanticMemory] Skipping stale: {doc[:50]}")
                    continue

            # Tool filter
            if tool_filter and meta.get("tool_name") != tool_filter:
                continue

            # Similarity filter
            similarity = 1.0 - (dist / 2.0)
            if similarity < SIMILARITY_THRESHOLD:
                continue

            results.append({
                "id":         rid,
                "text":       doc,
                "similarity": round(similarity, 3),
                "tool_name":  meta.get("tool_name", "unknown"),
                "query_used": meta.get("query_used", ""),
                "metadata":   meta,
            })

            if len(results) >= top_k:
                break

        if results:
            print(f"[SemanticMemory] Retrieved {len(results)} archives:")
            for r in results:
                print(f"  [{r['similarity']:.2f}] [{r['tool_name']}] {r['text'][:200]}")
        else:
            print("[SemanticMemory] No relevant archives found")

        return results

    # ── Stale management ────────────────────────────────────────

    def mark_stale_by_content(self, cancelled_value: str) -> int:
        """
        Mark entries containing any significant word from
        ``cancelled_value`` as stale.

        Mirrors EpisodicMemory.mark_stale_by_content() exactly.
        """
        total = self.collection.count()
        if total == 0:
            return 0

        words = cancelled_value.lower().split()
        significant = [
            w.strip(".,!?'\"-") for w in words
            if w.strip(".,!?'\"-") not in _STOP
            and len(w.strip(".,!?'\"-")) > 2
        ]
        if not significant:
            significant = [cancelled_value.lower()[:20]]

        print(
            f"[SemanticMemory] Marking stale entries "
            f"containing any of: {significant}"
        )

        try:
            all_entries = self.collection.get(include=["documents", "metadatas"])
        except Exception as exc:
            print(f"[SemanticMemory] get() error: {exc}")
            return 0

        docs  = all_entries.get("documents", [])
        metas = all_entries.get("metadatas",  [])
        ids   = all_entries.get("ids",        [])

        stale_count = 0
        for doc, meta, eid in zip(docs, metas, ids):
            doc_lower = doc.lower()

            raw_stale = meta.get("stale", "false")
            already   = (
                raw_stale if isinstance(raw_stale, bool)
                else str(raw_stale).lower() == "true"
            )
            if already:
                continue

            if not any(w in doc_lower for w in significant):
                continue

            updated = dict(meta)
            updated["stale"]        = "true"
            updated["staled_at"]    = datetime.now().isoformat()
            updated["stale_reason"] = f"Cancelled: {cancelled_value}"

            try:
                self.collection.update(ids=[eid], metadatas=[updated])
                stale_count += 1
                print(f"[SemanticMemory] Marked stale: {doc[:60]}")
            except Exception as exc:
                print(f"[SemanticMemory] update() error on {eid}: {exc}")

        print(f"[SemanticMemory] Total marked stale: {stale_count}")
        return stale_count

    # ── Utility ────────────────────────────────────────────────

    def get_all_active(self) -> list:
        """Return all non-stale entries. Filters in Python."""
        if self.collection.count() == 0:
            return []
        try:
            raw = self.collection.get(include=["documents", "metadatas"])
        except Exception as exc:
            print(f"[SemanticMemory] get_all_active error: {exc}")
            return []

        return [
            {"id": eid, "text": doc, "tool_name": meta.get("tool_name"), "metadata": meta}
            for doc, meta, eid in zip(
                raw.get("documents", []),
                raw.get("metadatas",  []),
                raw.get("ids",        []),
            )
            if not (
                meta.get("stale") if isinstance(meta.get("stale"), bool)
                else str(meta.get("stale", "false")).lower() == "true"
            )
        ]

    def get_count(self) -> dict:
        """Return {total, active, stale}."""
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
        """Delete the entire collection and recreate fresh."""
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print("[SemanticMemory] Reset complete")