# ccm/retriever.py
#
# Unified RAG retrieval across Tier 2 (Episodic) and Tier 3 (Semantic).
#
# TWO-STAGE RETRIEVAL:
#
#   Stage 1 — Vector Similarity Search (always)
#     ChromaDB finds entries whose embeddings are close to the query.
#     Fast. May include loosely related results.
#
#   Stage 2 — LLM Re-ranking (optional, use_reranking=True)
#     For each retrieved item, the LLM scores 0–3:
#       3 = Essential: agent NEEDS this
#       2 = Useful:    improves response quality
#       1 = Marginal:  probably not needed
#       0 = Irrelevant: discard
#     Only items scoring ≥ 2 are kept.
#     Slower but more precise. Disable for quick demos.
#
# TOKEN BUDGET:
#   Total retrieval budget: ~1100 tokens
#   Episodic gets half, Semantic gets half.
#   If budget is exceeded, entries are dropped last-first.

import os
import json
from groq import Groq
from dotenv import load_dotenv
from ccm.episodic_memory import EpisodicMemory
from ccm.semantic_memory  import SemanticMemory
from travel_agent.prompts import RETRIEVAL_RELEVANCE_PROMPT

load_dotenv()

MAX_RETRIEVAL_TOKENS = 1100
CHARS_PER_TOKEN      = 4    # rough estimate


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


class Retriever:
    """
    Unified RAG engine for Tier 2 + Tier 3.

    Public API:
      retrieve(query, n_episodic, n_semantic, token_budget) → dict
        Returns {episodic: list, semantic: list, total_tokens: int, query: str}
    """

    def __init__(
        self,
        episodic_memory: EpisodicMemory,
        semantic_memory: SemanticMemory,
        use_reranking: bool = True,
    ):
        self.episodic      = episodic_memory
        self.semantic      = semantic_memory
        self.use_reranking = use_reranking

        if use_reranking:
            self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            self.model  = "llama-3.1-8b-instant"

    def retrieve(
        self,
        query: str,
        n_episodic: int = 4,
        n_semantic:  int = 3,
        token_budget: int = MAX_RETRIEVAL_TOKENS,
    ) -> dict:
        """
        Search both memory tiers and return relevant results.

        Parameters
        ----------
        query        : str   Current user message (used as search query)
        n_episodic   : int   Max episodic results to fetch initially
        n_semantic   : int   Max semantic results to fetch initially
        token_budget : int   Total token cap for all results combined

        Returns
        -------
        dict  {episodic, semantic, total_tokens, query}
        """
        print(f"\n[Retriever] Query: '{query[:60]}'")

        # Stage 1: vector search
        raw_ep  = self._search_episodic(query, n_episodic)
        raw_sem = self._search_semantic(query, n_semantic)
        print(
            f"[Retriever] Stage 1: {len(raw_ep)} episodic, "
            f"{len(raw_sem)} semantic"
        )

        # Stage 2: optional LLM re-ranking
        if self.use_reranking and (raw_ep or raw_sem):
            all_raw      = raw_ep + raw_sem
            all_reranked = self._rerank(query, all_raw)

            ep_ids  = {r["id"] for r in raw_ep}
            final_ep  = [r for r in all_reranked if r["id"] in ep_ids]
            final_sem = [r for r in all_reranked if r["id"] not in ep_ids]
            print(
                f"[Retriever] After re-rank: {len(final_ep)} episodic, "
                f"{len(final_sem)} semantic"
            )
        else:
            final_ep  = raw_ep
            final_sem = raw_sem

        # Apply token budget
        final_ep, final_sem, total_tokens = self._apply_budget(
            final_ep, final_sem, token_budget
        )
        print(
            f"[Retriever] Final: {len(final_ep)} episodic, "
            f"{len(final_sem)} semantic, ~{total_tokens} tokens"
        )

        return {
            "episodic":     final_ep,
            "semantic":     final_sem,
            "total_tokens": total_tokens,
            "query":        query,
        }

    # ── Stage 1 helpers ─────────────────────────────────────────

    def _search_episodic(self, query: str, top_k: int) -> list:
        try:
            return self.episodic.retrieve(query=query, top_k=top_k, exclude_stale=True)
        except Exception as exc:
            print(f"[Retriever] Episodic search error: {exc}")
            return []

    def _search_semantic(self, query: str, top_k: int) -> list:
        try:
            return self.semantic.retrieve(query=query, top_k=top_k, exclude_stale=True)
        except Exception as exc:
            print(f"[Retriever] Semantic search error: {exc}")
            return []

    # ── Public separate retrieval methods ─────────────────────────────────

    def retrieve_episodic_only(self, query: str, n_results: int = 4) -> list:
        """Retrieve only from episodic memory (always called)."""
        raw_ep = self._search_episodic(query, n_results)
        if self.use_reranking and raw_ep:
            raw_ep = self._rerank_episodic_only(query, raw_ep)

        # Apply budget to episodic
        final_ep = []
        total = 0
        for r in raw_ep:
            t = _estimate_tokens(r["text"])
            if total + t <= MAX_RETRIEVAL_TOKENS // 2:
                final_ep.append(r)
                total += t
            else:
                break
        return final_ep

    def retrieve_semantic_only(self, query: str, n_results: int = 3) -> list:
        """Retrieve only from semantic memory (called when query references past)."""
        raw_sem = self._search_semantic(query, n_results)
        if self.use_reranking and raw_sem:
            raw_sem = self._rerank_semantic_only(query, raw_sem)

        # Apply budget to semantic
        final_sem = []
        for r in raw_sem:
            t = _estimate_tokens(r["text"])
            if t <= MAX_RETRIEVAL_TOKENS:
                final_sem.append(r)
            else:
                break
        return final_sem

    def _rerank_episodic_only(self, query: str, results: list) -> list:
        """Re-rank only episodic results."""
        if not results:
            return []
        # Simplified re-ranking for episodic only
        return results[:4]

    def _rerank_semantic_only(self, query: str, results: list) -> list:
        """Re-rank only semantic results."""
        if not results:
            return []
        # Simplified re-ranking for semantic only
        return results[:3]

    # ── Stage 2 re-ranking ───────────────────────────────────────

    def _rerank(self, query: str, results: list) -> list:
        """
        Ask the LLM to score each result 0–3.
        Keep only items scoring ≥ 2.
        Falls back to raw results on any error.
        """
        if not results:
            return []

        items_text = ""
        for r in results:
            items_text += (
                f"ID: {r['id']}\n"
                f"Text: {r['text']}\n"
                f"Similarity: {r.get('similarity', '?')}\n\n"
            )

        prompt = RETRIEVAL_RELEVANCE_PROMPT.format(
            query=query,
            retrieved_items=items_text,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Score retrieved memory items for relevance. "
                            "Return ONLY valid JSON with a 'scores' array. "
                            "Each element: {id, score (0-3), reason}."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=500,
                temperature=0.0,
            )

            raw = response.choices[0].message.content.strip()
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()

            start, end = raw.find("{"), raw.rfind("}") + 1
            if start >= 0 and end > start:
                raw = raw[start:end]

            data   = json.loads(raw)
            scores = {
                s["id"]: s.get("score", 1)
                for s in data.get("scores", [])
            }

            reranked = []
            for r in results:
                score = scores.get(r["id"], 1)
                if score >= 2:
                    rc = dict(r)
                    rc["relevance_score"] = score
                    reranked.append(rc)
                else:
                    print(
                        f"[Retriever] Dropped (score={score}): "
                        f"{r['text'][:50]}"
                    )

            reranked.sort(
                key=lambda x: (x.get("relevance_score", 0), x.get("similarity", 0)),
                reverse=True,
            )
            print(
                f"[Retriever] Re-ranking: {len(results)} → "
                f"{len(reranked)} kept"
            )
            return reranked

        except Exception as exc:
            print(f"[Retriever] Re-rank error ({exc}) — using raw results")
            return results

    # ── Token budget ─────────────────────────────────────────────

    def _apply_budget(
        self, episodic: list, semantic: list, budget: int
    ) -> tuple:
        """Trim results to stay within token budget."""
        total  = 0
        final_ep  = []
        final_sem = []

        for r in episodic:
            t = _estimate_tokens(r["text"])
            if total + t <= budget:
                final_ep.append(r)
                total += t
            else:
                print("[Retriever] Budget hit — dropping episodic entry")
                break

        for r in semantic:
            t = _estimate_tokens(r["text"])
            if total + t <= budget:
                final_sem.append(r)
                total += t
            else:
                print("[Retriever] Budget hit — dropping semantic entry")
                break

        return final_ep, final_sem, total