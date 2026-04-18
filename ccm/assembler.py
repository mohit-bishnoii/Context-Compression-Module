# ccm/assembler.py
#
# Builds the final compressed context packet injected before every LLM call.
#
# WHAT IT PRODUCES:
#   ──────────────────────────────────────────────────
#   [CRITICAL CONSTRAINTS AND PREFERENCES]
#   [CRITICAL CONSTRAINTS]
#     • severely allergic to shellfish
#   [USER PREFERENCES]
#     • maximum 2 activities per day
#
#   ──────────────────────────────────────────────────
#   [RELEVANT CONVERSATION HISTORY]
#     • Searched flights JFK→NRT. ANA $780 direct approved.
#
#   ──────────────────────────────────────────────────
#   [RELEVANT RESEARCH AND DETAILS]
#     [places_search] Tsukiji: Sushi Dai AVOID (shellfish)…
#
#   ──────────────────────────────────────────────────
#   [RECENT CONVERSATION]
#     USER: find restaurants near Tsukiji
#     ASSISTANT: …
#   ──────────────────────────────────────────────────
#
# TOKEN BUDGET:
#   Total target : 2000 tokens
#   Working mem  : ~400  (always present)
#   Recent turns : ~600  (always present)
#   Retrieved    : ~1000 (RAG-selected)

import tiktoken
from travel_agent.prompts import (
    SECTION_WORKING_MEMORY,
    SECTION_EPISODIC,
    SECTION_ARCHIVED,
    SECTION_RECENT,
    SECTION_DIVIDER,
)

TOTAL_CONTEXT_BUDGET  = 2000
RECENT_TURNS_BUDGET   = 600
WORKING_MEM_BUDGET    = 400
RETRIEVAL_BUDGET      = TOTAL_CONTEXT_BUDGET - RECENT_TURNS_BUDGET - WORKING_MEM_BUDGET


def _count_tokens(text: str) -> int:
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


class ContextAssembler:
    """
    Assembles the compressed context packet for each LLM call.

    Public API:
      assemble(working_memory, retrieved, recent_turns,
               max_recent_turns=3)          → str
      get_last_token_count()                → int
      get_breakdown()                       → dict
      format_for_display(...)               → dict  (for UI)
    """

    def __init__(self):
        self.last_token_count       = 0
        self.last_assembly_breakdown = {}

    def assemble(
        self,
        working_memory,
        retrieved: dict,
        recent_turns: list,
        max_recent_turns: int = 3,
    ) -> str:
        """
        Build the full context packet string.

        Parameters
        ----------
        working_memory   : WorkingMemory instance
        retrieved        : Output from Retriever.retrieve()
                           {episodic: [...], semantic: [...]}
        recent_turns     : Full conversation history list
                           [{"role": "user/assistant", "content": "…"}]
        max_recent_turns : How many recent turns to include verbatim

        Returns
        -------
        str  The assembled context string to prepend to the user message.
        """
        sections        = []
        token_breakdown = {}

        # ── Section 1: Working Memory (ALWAYS present) ───────────
        wm_text   = working_memory.format_for_prompt()
        wm_tokens = _count_tokens(wm_text)

        sections.append(SECTION_DIVIDER)
        sections.append(SECTION_WORKING_MEMORY.strip())
        sections.append(wm_text)
        token_breakdown["working_memory"] = wm_tokens

        # ── Section 2: Episodic memories (RAG) ──────────────────
        episodic_results = retrieved.get("episodic", [])
        if episodic_results:
            lines  = []
            tokens = 0
            for r in episodic_results:
                line       = f"  • {r['text']}"
                line_tokens = _count_tokens(line)
                if tokens + line_tokens > RETRIEVAL_BUDGET // 2:
                    break
                lines.append(line)
                tokens += line_tokens
            if lines:
                sections.append(SECTION_DIVIDER)
                sections.append(SECTION_EPISODIC.strip())
                sections.extend(lines)
                token_breakdown["episodic"] = tokens

        # ── Section 3: Semantic / archived memories (RAG) ────────
        semantic_results = retrieved.get("semantic", [])
        if semantic_results:
            lines  = []
            tokens = 0
            for r in semantic_results:
                tool = r.get("tool_name", "tool")
                line = f"  [{tool}] {r['text']}"
                line_tokens = _count_tokens(line)
                if tokens + line_tokens > RETRIEVAL_BUDGET // 2:
                    break
                lines.append(line)
                tokens += line_tokens
            if lines:
                sections.append(SECTION_DIVIDER)
                sections.append(SECTION_ARCHIVED.strip())
                sections.extend(lines)
                token_breakdown["semantic"] = tokens

        # ── Section 4: Recent conversation turns (ALWAYS) ────────
        # Take the last max_recent_turns * 2 messages
        recent = recent_turns[-(max_recent_turns * 2):] if recent_turns else []
        if recent:
            lines  = []
            tokens = 0
            for turn in recent:
                role    = turn.get("role", "unknown").upper()
                content = turn.get("content", "")

                if role == "TOOL":      # skip raw tool call messages
                    continue

                if len(content) > 500:  # truncate very long messages
                    content = content[:500] + "…[truncated]"

                line       = f"  {role}: {content}"
                line_tokens = _count_tokens(line)
                if tokens + line_tokens > RECENT_TURNS_BUDGET:
                    break
                lines.append(line)
                tokens += line_tokens

            if lines:
                sections.append(SECTION_DIVIDER)
                sections.append(SECTION_RECENT.strip())
                sections.extend(lines)
                token_breakdown["recent_turns"] = tokens

        sections.append(SECTION_DIVIDER)

        assembled = "\n".join(sections)

        self.last_token_count        = _count_tokens(assembled)
        self.last_assembly_breakdown = token_breakdown

        print(
            f"[Assembler] Context assembled: {self.last_token_count} tokens"
        )
        print(f"  Breakdown: {token_breakdown}")

        return assembled

    def get_last_token_count(self) -> int:
        return self.last_token_count

    def get_breakdown(self) -> dict:
        return self.last_assembly_breakdown

    def format_for_display(
        self,
        working_memory,
        retrieved: dict,
        recent_turns: list,
    ) -> dict:
        """Format context info for UI display (Gradio memory panel)."""
        return {
            "working_memory": working_memory.format_for_prompt(),
            "episodic":  [r["text"] for r in retrieved.get("episodic", [])],
            "semantic":  [
                f"[{r.get('tool_name', 'tool')}] {r['text']}"
                for r in retrieved.get("semantic", [])
            ],
            "recent_turns": recent_turns[-6:],
            "total_tokens": self.last_token_count,
            "breakdown":    self.last_assembly_breakdown,
        }