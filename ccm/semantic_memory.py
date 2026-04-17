# ccm/semantic_memory.py
#
# Tier 3: Semantic / Archived Memory
#
# Stores compressed tool results for RAG retrieval.
# Completely rewritten for reliability:
#   - Stale stored as string "true"/"false" (ChromaDB bool bug fix)
#   - Filtering done in Python not ChromaDB where clause
#   - Supports both persistent (production) and in-memory (testing)
#   - Clear logging for every operation

import os
import uuid
from datetime import datetime
from typing import Optional

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

import chromadb
from ccm.episodic_memory import embed

CHROMA_PATH = "./data/chroma_db"
COLLECTION_NAME = "semantic_memory"
DEFAULT_TOP_K = 3
SIMILARITY_THRESHOLD = 0.3


class SemanticMemory:
    """
    Tier 3: Semantic / Archived Memory

    Stores compressed tool results.
    Retrieved via vector similarity search (RAG).

    Key design decisions:
    - Stale stored as string "true"/"false" not boolean
      because ChromaDB metadata boolean filtering is
      unreliable across versions
    - Stale filtering happens in Python after fetch
    - Supports in_memory mode for testing (no file locks)

    Agent-centric: stores any tool result from any domain.
    Does not know about travel specifically.
    """

    def __init__(self, in_memory: bool = False):
        """
        Parameters:
          in_memory: True  = RAM only, no files (use for testing)
                     False = persists to disk (use in production)
        """
        os.makedirs(CHROMA_PATH, exist_ok=True)
        os.makedirs("data", exist_ok=True)

        if in_memory:
            self.client = chromadb.EphemeralClient()
            self._mode = "in-memory"
        else:
            self.client = chromadb.PersistentClient(path=CHROMA_PATH)
            self._mode = "persistent"

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

        print(
            f"[SemanticMemory] Ready ({self._mode}). "
            f"Contains {self.collection.count()} entries."
        )

    # ── Write Operations ────────────────────────────────────────

    def add(
        self,
        compressed_result: str,
        tool_name: str,
        query_used: str,
        turn_number: int = 0,
        metadata: Optional[dict] = None
    ) -> str:
        """
        Store a compressed tool result.

        The text embedded is query + result combined.
        This makes retrieval work better because future queries
        about the same topic find this entry more reliably.

        Parameters:
          compressed_result: Already-compressed tool output string
          tool_name:         Name of tool that produced this
          query_used:        Query passed to the tool
          turn_number:       Conversation turn number
          metadata:          Optional extra metadata

        Returns:
          Unique ID string or empty string if failed
        """
        if not compressed_result or not compressed_result.strip():
            print("[SemanticMemory] Empty result, skipping")
            return ""

        memory_id = f"sem_{uuid.uuid4().hex[:12]}"

        # Embed query + result together
        # This improves retrieval because the embedding
        # captures both the question and the answer context
        embeddable_text = (
            f"Query: {query_used}\n"
            f"Result: {compressed_result}"
        )

        entry_metadata = {
            "tool_name": tool_name,
            "query_used": query_used,
            "turn_number": turn_number,
            "created_at": datetime.now().isoformat(),
            "stale": "false",   # STRING not bool
            "type": "tool_result"
        }

        if metadata:
            # Ensure no boolean values in metadata
            for k, v in metadata.items():
                if isinstance(v, bool):
                    metadata[k] = "true" if v else "false"
            entry_metadata.update(metadata)

        try:
            vector = embed(embeddable_text)
            self.collection.add(
                documents=[compressed_result],
                embeddings=[vector],
                metadatas=[entry_metadata],
                ids=[memory_id]
            )
            print(
                f"[SemanticMemory] Stored {tool_name} result: "
                f"{memory_id}"
            )
            print(f"  Content: {compressed_result[:80]}...")
            return memory_id

        except Exception as e:
            print(f"[SemanticMemory] Add error: {e}")
            return ""

    # ── Read Operations ─────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        tool_filter: Optional[str] = None,
        exclude_stale: bool = True
    ) -> list:
        """
        Find relevant archived tool results for a query.

        HOW IT WORKS:
          1. Embed the query into a vector
          2. Fetch top_k * 3 results by vector similarity
             (fetch more so we have room to filter)
          3. Filter stale entries in Python (not ChromaDB)
          4. Apply tool_filter if specified
          5. Return top_k results

        Parameters:
          query:        Current user message or search topic
          top_k:        Maximum results to return
          tool_filter:  If set, only return results from this tool
          exclude_stale: Skip stale entries

        Returns:
          List of dicts:
          {id, text, similarity, tool_name, query_used, metadata}
        """
        total = self.collection.count()
        if total == 0:
            print("[SemanticMemory] Collection is empty")
            return []

        # Fetch more than needed to allow for filtering
        fetch_k = min(top_k * 3, total)

        try:
            query_vector = embed(query)
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=fetch_k,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            print(f"[SemanticMemory] Query error: {e}")
            return []

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        if not documents:
            print("[SemanticMemory] No results from ChromaDB")
            return []

        retrieved = []

        for doc, meta, dist, rid in zip(
            documents, metadatas, distances, ids
        ):
            # ── Stale filter (Python-side) ───────────────────────
            if exclude_stale:
                stale_raw = meta.get("stale", "false")
                # Handle bool True/False AND string "true"/"false"
                if isinstance(stale_raw, bool):
                    is_stale = stale_raw
                elif isinstance(stale_raw, str):
                    is_stale = stale_raw.lower() == "true"
                else:
                    is_stale = False

                if is_stale:
                    print(
                        f"[SemanticMemory] Skipping stale: "
                        f"{doc[:40]}..."
                    )
                    continue

            # ── Tool filter ──────────────────────────────────────
            if tool_filter:
                if meta.get("tool_name") != tool_filter:
                    continue

            # ── Similarity threshold ─────────────────────────────
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity: 1 - (distance / 2)
            similarity = 1.0 - (dist / 2.0)
            if similarity < SIMILARITY_THRESHOLD:
                continue

            retrieved.append({
                "id": rid,
                "text": doc,
                "similarity": round(similarity, 3),
                "tool_name": meta.get("tool_name", "unknown"),
                "query_used": meta.get("query_used", ""),
                "metadata": meta
            })

            # Stop once we have enough
            if len(retrieved) >= top_k:
                break

        if retrieved:
            print(
                f"[SemanticMemory] Retrieved "
                f"{len(retrieved)} archived results"
            )
            for r in retrieved:
                print(
                    f"  [{r['similarity']:.2f}] "
                    f"[{r['tool_name']}] {r['text'][:60]}..."
                )
        else:
            print("[SemanticMemory] No relevant archives found")

        return retrieved

    # ── Stale Operations ────────────────────────────────────────

    def mark_stale_by_content(self, substring: str) -> int:
        """
        Mark entries containing any significant word from
        substring as stale. They will be skipped in retrieval.

        Uses word-level matching, not full phrase matching,
        because stored text may use different phrasing than
        the cancelled value.

        Example:
          substring = "Bali beach vacation"
          significant words = ["bali", "beach"]
          Matches: "Researched Bali resorts" ← contains "bali"
          Does not match: "Tokyo hotels" ← no match

        Returns:
          Number of entries marked stale
        """
        total = self.collection.count()
        if total == 0:
            return 0

        # Extract significant words
        stop_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were',
            'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'trip',
            'plan', 'planned', 'vacation', 'holiday', 'let',
            'do', 'us', 'me', 'my', 'its', 'this', 'that',
            'have', 'has', 'will', 'would', 'could', 'should'
        }

        words = substring.lower().split()
        significant = [
            w.strip('.,!?\'\"')
            for w in words
            if w.strip('.,!?\'\"') not in stop_words
            and len(w.strip('.,!?\'\"')) > 3
        ]

        if not significant:
            significant = [substring.lower()[:20]]

        print(
            f"[SemanticMemory] Looking for stale entries "
            f"containing: {significant}"
        )

        try:
            all_entries = self.collection.get(
                include=["documents", "metadatas"]
            )
        except Exception as e:
            print(f"[SemanticMemory] Get error: {e}")
            return 0

        documents = all_entries.get("documents", [])
        metadatas = all_entries.get("metadatas", [])
        ids = all_entries.get("ids", [])

        stale_count = 0

        for doc, meta, entry_id in zip(documents, metadatas, ids):
            doc_lower = doc.lower()

            # Check if any significant word appears in document
            if not any(word in doc_lower for word in significant):
                continue

            # Skip if already stale
            stale_raw = meta.get("stale", "false")
            if isinstance(stale_raw, bool):
                already_stale = stale_raw
            else:
                already_stale = str(stale_raw).lower() == "true"

            if already_stale:
                continue

            # Mark as stale
            updated_meta = dict(meta)
            updated_meta["stale"] = "true"  # Always string
            updated_meta["staled_at"] = datetime.now().isoformat()
            updated_meta["stale_reason"] = f"Cancelled: {substring}"

            try:
                self.collection.update(
                    ids=[entry_id],
                    metadatas=[updated_meta]
                )
                stale_count += 1
                print(
                    f"[SemanticMemory] Marked stale: "
                    f"{doc[:60]}..."
                )
            except Exception as e:
                print(f"[SemanticMemory] Update error: {e}")

        if stale_count > 0:
            print(
                f"[SemanticMemory] Total marked stale: "
                f"{stale_count}"
            )
        else:
            print(
                f"[SemanticMemory] No entries found "
                f"containing {significant}"
            )

        return stale_count

    # ── Utility Operations ───────────────────────────────────────

    def get_all_active(self) -> list:
        """
        Get all non-stale entries.
        Used by Gradio UI to display memory state.
        """
        total = self.collection.count()
        if total == 0:
            return []

        try:
            results = self.collection.get(
                include=["documents", "metadatas"]
            )
        except Exception as e:
            print(f"[SemanticMemory] Get error: {e}")
            return []

        entries = []
        for doc, meta, entry_id in zip(
            results.get("documents", []),
            results.get("metadatas", []),
            results.get("ids", [])
        ):
            stale_raw = meta.get("stale", "false")
            if isinstance(stale_raw, bool):
                is_stale = stale_raw
            else:
                is_stale = str(stale_raw).lower() == "true"

            if not is_stale:
                entries.append({
                    "id": entry_id,
                    "text": doc,
                    "tool_name": meta.get("tool_name", "unknown"),
                    "metadata": meta
                })

        return entries

    def get_count(self) -> dict:
        """Return count statistics."""
        total = self.collection.count()
        if total == 0:
            return {"total": 0, "active": 0, "stale": 0}

        try:
            all_meta = self.collection.get(
                include=["metadatas"]
            ).get("metadatas", [])
        except Exception:
            return {"total": total, "active": total, "stale": 0}

        stale_count = 0
        for meta in all_meta:
            stale_raw = meta.get("stale", "false")
            if isinstance(stale_raw, bool):
                if stale_raw:
                    stale_count += 1
            elif str(stale_raw).lower() == "true":
                stale_count += 1

        return {
            "total": total,
            "active": total - stale_count,
            "stale": stale_count
        }

    def reset(self):
        """Delete all entries. Called on new conversation."""
        try:
            self.client.delete_collection(COLLECTION_NAME)
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            print("[SemanticMemory] Reset complete")
        except Exception as e:
            print(f"[SemanticMemory] Reset error: {e}")