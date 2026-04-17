# ccm/retriever.py
#
# Unified RAG retrieval across Tier 2 and Tier 3 memory.
#
# WHAT THIS DOES:
#   Takes the current user query.
#   Searches episodic memory (conversation summaries).
#   Searches semantic memory (compressed tool results).
#   Optionally re-ranks results using LLM scoring.
#   Returns results within a token budget.
#
# WHY THIS IS NOT SIMPLE RAG:
#   Simple RAG: search static documents, return top N
#   Our system:
#     Searches DYNAMIC memories created from conversation
#     Searches TWO separate memory tiers with different purposes
#     Filters stale/cancelled entries automatically
#     Optional LLM re-ranking for better precision
#     Token-budget aware — never overflows context window
#
# AGENT-CENTRIC:
#   This class only knows about episodic and semantic memory.
#   It does not know what the memories contain.
#   Works for travel, medical, legal, or any other agent.

import os
import json

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

from groq import Groq
from dotenv import load_dotenv
from ccm.episodic_memory import EpisodicMemory
from ccm.semantic_memory import SemanticMemory
from travel_agent.prompts import RETRIEVAL_RELEVANCE_PROMPT

load_dotenv()

# Token budget for all retrieved content combined
# Total context budget: ~2000 tokens
# Working memory uses: ~300 tokens
# Recent turns use:    ~600 tokens
# Available for retrieval: ~1100 tokens
MAX_RETRIEVAL_TOKENS = 1100

# Rough token estimation: 1 token ≈ 4 characters
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Fast token estimation without tiktoken."""
    return max(1, len(text) // CHARS_PER_TOKEN)


class Retriever:
    """
    Unified RAG retrieval engine.

    Two-stage retrieval process:

    STAGE 1 — Vector Similarity Search (always runs)
      ChromaDB finds memories with similar embeddings.
      Fast. May return loosely related results.

    STAGE 2 — LLM Re-ranking (optional, use_reranking=True)
      LLM scores each result for actual usefulness.
      Slower. More precise. Filters out false positives.
      This is what makes our RAG better than simple RAG.

    For demos: use_reranking=False (faster)
    For evaluation: use_reranking=True (more accurate)
    """

    def __init__(
        self,
        episodic_memory: EpisodicMemory,
        semantic_memory: SemanticMemory,
        use_reranking: bool = True
    ):
        self.episodic = episodic_memory
        self.semantic = semantic_memory
        self.use_reranking = use_reranking

        if use_reranking:
            self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            self.model = "llama-3.1-8b-instant"

    # ── Main Retrieval ──────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        n_episodic: int = 4,
        n_semantic: int = 3,
        token_budget: int = MAX_RETRIEVAL_TOKENS
    ) -> dict:
        """
        Search both memory tiers and return relevant results.

        Parameters:
          query:        Current user message
          n_episodic:   Max episodic results to fetch initially
          n_semantic:   Max semantic results to fetch initially
          token_budget: Max tokens the results can use combined

        Returns:
          {
            "episodic": list of relevant episodic memories,
            "semantic": list of relevant semantic memories,
            "total_tokens": estimated tokens used,
            "query": the query used
          }
        """
        print(f"\n[Retriever] Query: '{query[:60]}'")

        # ── Stage 1: Vector Similarity Search ───────────────────
        raw_episodic = self._search_episodic(query, n_episodic)
        raw_semantic = self._search_semantic(query, n_semantic)

        print(
            f"[Retriever] Stage 1: "
            f"{len(raw_episodic)} episodic, "
            f"{len(raw_semantic)} semantic"
        )

        # ── Stage 2: LLM Re-ranking (optional) ──────────────────
        if self.use_reranking and (raw_episodic or raw_semantic):
            all_raw = raw_episodic + raw_semantic
            all_reranked = self._rerank(query, all_raw)

            # Split back by source
            episodic_ids = {r["id"] for r in raw_episodic}
            final_episodic = [
                r for r in all_reranked
                if r["id"] in episodic_ids
            ]
            final_semantic = [
                r for r in all_reranked
                if r["id"] not in episodic_ids
            ]

            print(
                f"[Retriever] After re-ranking: "
                f"{len(final_episodic)} episodic, "
                f"{len(final_semantic)} semantic"
            )
        else:
            final_episodic = raw_episodic
            final_semantic = raw_semantic

        # ── Apply Token Budget ───────────────────────────────────
        final_episodic, final_semantic, total_tokens = (
            self._apply_budget(
                final_episodic,
                final_semantic,
                token_budget
            )
        )

        print(
            f"[Retriever] Final: "
            f"{len(final_episodic)} episodic, "
            f"{len(final_semantic)} semantic, "
            f"~{total_tokens} tokens"
        )

        return {
            "episodic": final_episodic,
            "semantic": final_semantic,
            "total_tokens": total_tokens,
            "query": query
        }

    # ── Stage 1: Vector Search ───────────────────────────────────

    def _search_episodic(self, query: str, top_k: int) -> list:
        """Search episodic memory. Returns empty list on any error."""
        try:
            return self.episodic.retrieve(
                query=query,
                top_k=top_k,
                exclude_stale=True
            )
        except Exception as e:
            print(f"[Retriever] Episodic search error: {e}")
            return []

    def _search_semantic(self, query: str, top_k: int) -> list:
        """Search semantic memory. Returns empty list on any error."""
        try:
            return self.semantic.retrieve(
                query=query,
                top_k=top_k,
                exclude_stale=True
            )
        except Exception as e:
            print(f"[Retriever] Semantic search error: {e}")
            return []

    # ── Stage 2: LLM Re-ranking ──────────────────────────────────

    def _rerank(self, query: str, results: list) -> list:
        """
        Score each retrieved result for actual relevance.

        Scores: 0=irrelevant, 1=marginal, 2=useful, 3=essential
        Keeps: score >= 2
        Sorts: by score descending

        If re-ranking fails for any reason, returns
        original results unchanged (graceful degradation).
        """
        if not results:
            return []

        # Build items text for the prompt
        items_text = ""
        for r in results:
            items_text += (
                f"ID: {r['id']}\n"
                f"Text: {r['text']}\n"
                f"Similarity: {r.get('similarity', '?')}\n\n"
            )

        prompt = RETRIEVAL_RELEVANCE_PROMPT.format(
            query=query,
            retrieved_items=items_text
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You score retrieved memory items. "
                            "Return only valid JSON."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.0
            )

            raw = response.choices[0].message.content.strip()

            # Strip markdown if present
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()

            # Find JSON object
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                raw = raw[start:end]

            data = json.loads(raw)
            scores = {
                s["id"]: s.get("score", 1)
                for s in data.get("scores", [])
            }

            # Filter by score threshold and sort
            reranked = []
            for result in results:
                score = scores.get(result["id"], 1)
                if score >= 2:
                    result_copy = dict(result)
                    result_copy["relevance_score"] = score
                    reranked.append(result_copy)
                else:
                    print(
                        f"[Retriever] Dropped (score={score}): "
                        f"{result['text'][:50]}..."
                    )

            # Sort best first
            reranked.sort(
                key=lambda x: (
                    x.get("relevance_score", 0),
                    x.get("similarity", 0)
                ),
                reverse=True
            )

            print(
                f"[Retriever] Re-ranking: "
                f"{len(results)} → {len(reranked)} kept"
            )
            return reranked

        except json.JSONDecodeError as e:
            print(f"[Retriever] Re-rank JSON error: {e}. Using raw.")
            return results
        except Exception as e:
            print(f"[Retriever] Re-rank error: {e}. Using raw.")
            return results

    # ── Token Budget ─────────────────────────────────────────────

    def _apply_budget(
        self,
        episodic: list,
        semantic: list,
        budget: int
    ) -> tuple:
        """
        Trim results to fit within token budget.

        Priority: episodic first (conversation context),
                  semantic second (research details).

        Returns: (trimmed_episodic, trimmed_semantic, tokens_used)
        """
        total_tokens = 0
        final_episodic = []
        final_semantic = []

        for result in episodic:
            tokens = estimate_tokens(result["text"])
            if total_tokens + tokens <= budget:
                final_episodic.append(result)
                total_tokens += tokens
            else:
                print(
                    f"[Retriever] Budget limit hit, "
                    f"dropping episodic result"
                )
                break

        for result in semantic:
            tokens = estimate_tokens(result["text"])
            if total_tokens + tokens <= budget:
                final_semantic.append(result)
                total_tokens += tokens
            else:
                print(
                    f"[Retriever] Budget limit hit, "
                    f"dropping semantic result"
                )
                break

        return final_episodic, final_semantic, total_tokens