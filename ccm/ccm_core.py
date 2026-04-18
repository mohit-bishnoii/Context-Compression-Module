# ccm/ccm_core.py
#
# The Context Compression Module — the single entry point for agents.
#
# AGENT API (only three methods the agent ever calls):
#
#   ccm = ContextCompressionModule()
#
#   # BEFORE sending to LLM:
#   context = ccm.process_user_message(user_message)
#   → Extracts facts, detects stale, retrieves memories,
#     assembles ~1400-token context packet. Returns it as a string.
#
#   # AFTER getting a tool result:
#   compressed = ccm.process_tool_result(tool_name, raw_result, query)
#   → Compresses 600-token result to ~80 tokens.
#     Stores in SemanticMemory. Returns compressed string.
#
#   # AFTER agent responds:
#   ccm.process_agent_response(user_msg, agent_response, tool_calls)
#   → Appends to conversation history. Every 4 turns, creates
#     an episodic summary and stores in EpisodicMemory.
#
#   # For UI display:
#   state = ccm.get_memory_state()
#
# WHAT HAPPENS EACH TURN (detailed):
#
#   process_user_message(msg):
#     1. increment turn counter
#     2. Extractor: LLM call → extract facts → add to WorkingMemory
#     3. StaleDetector: if override signal → LLM call →
#        remove keys from WorkingMemory, mark stale in both
#        EpisodicMemory AND SemanticMemory
#     4. Retriever: embed query → ChromaDB search → optional re-rank
#     5. Assembler: build context packet (WorkingMemory always first)
#     6. Return context string
#
#   process_tool_result(tool, raw, query):
#     1. Get critical constraints from WorkingMemory
#     2. Compressor: LLM call → 600 tokens → ~80 tokens
#     3. Store compressed result in SemanticMemory
#     4. Return compressed string (agent uses this instead of raw)
#
#   process_agent_response(user, response, tool_calls):
#     1. Append user + assistant to conversation_history
#     2. Append to turn_buffer
#     3. If len(turn_buffer) >= 4:
#          LLM call → episodic summary → EpisodicMemory.add()
#          Clear buffer

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"]     = "False"

import json
import tiktoken
from dotenv import load_dotenv

from ccm.memory_store    import WorkingMemory
from ccm.episodic_memory import EpisodicMemory
from ccm.semantic_memory import SemanticMemory
from ccm.extractor       import MemoryExtractor
from ccm.compressor      import ToolCompressor
from ccm.stale_detector  import StaleDetector
from ccm.retriever       import Retriever
from ccm.assembler       import ContextAssembler

load_dotenv()

EPISODE_EVERY_N_TURNS = 4


def _count_tokens(text: str) -> int:
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


class ContextCompressionModule:
    """
    The Context Compression Module.

    Drop this between any user interface and any LLM agent.
    The agent calls three methods; everything else is internal.

    Parameters
    ----------
    use_reranking : bool
        If True, uses LLM re-ranking in retrieval (slower, more precise).
        Set False for demos/tests to reduce API calls.
    """

    def __init__(self, use_reranking: bool = True):
        print("[CCM] Initialising Context Compression Module…")
        os.makedirs("data", exist_ok=True)
        os.makedirs("data/chroma_db", exist_ok=True)

        # Three memory tiers
        self.working_memory  = WorkingMemory()
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()

        # Processing components
        self.extractor    = MemoryExtractor()
        self.compressor   = ToolCompressor()
        self.stale_detector = StaleDetector()

        # Retrieval and assembly
        self.retriever = Retriever(
            episodic_memory=self.episodic_memory,
            semantic_memory=self.semantic_memory,
            use_reranking=use_reranking,
        )
        self.assembler = ContextAssembler()

        # Conversation state
        self.conversation_history  = []   # full raw history
        self.turn_buffer           = []   # turns since last episode
        self.turn_count            = 0
        self.total_baseline_tokens = 0
        self.total_ccm_tokens      = 0

        print("[CCM] Ready")

    # ── Public API ───────────────────────────────────────────────

    def process_user_message(self, user_message: str) -> str:
        """
        Process a user message and return the compressed context string.

        Call this BEFORE every LLM call.
        Prepend the returned string to the user message in your prompt.
        """
        self.turn_count += 1
        self.working_memory.increment_turn()

        print(f"\n{'='*50}")
        print(f"[CCM] Turn {self.turn_count}")
        print(f"{'='*50}")

        # Step 1: Extract facts
        print("[CCM] Step 1: Extracting facts…")
        try:
            self.extractor.extract_and_update(user_message, self.working_memory)
        except Exception as exc:
            print(f"[CCM] Extractor error (non-fatal): {exc}")

        # Step 2: Detect and clean stale context
        print("[CCM] Step 2: Checking stale context…")
        try:
            stale = self.stale_detector.check_and_clean(
                user_message,
                self.working_memory,
                self.episodic_memory,
                self.semantic_memory,   # ← BOTH tiers passed
            )
            if stale.get("has_override"):
                print(f"[CCM] Cleaned stale: {stale.get('cancelled_values')}")
        except Exception as exc:
            print(f"[CCM] StaleDetector error (non-fatal): {exc}")

        # Step 3: Retrieve relevant memories
        print("[CCM] Step 3: Retrieving memories…")
        try:
            retrieved = self.retriever.retrieve(
                query=user_message,
                n_episodic=4,
                n_semantic=3,
            )
        except Exception as exc:
            print(f"[CCM] Retriever error (non-fatal): {exc}")
            retrieved = {"episodic": [], "semantic": []}

        # Step 4: Assemble context packet
        print("[CCM] Step 4: Assembling context…")
        context = self.assembler.assemble(
            working_memory=self.working_memory,
            retrieved=retrieved,
            recent_turns=self.conversation_history,
            max_recent_turns=3,
        )

        # Track metrics
        ccm_tokens       = self.assembler.get_last_token_count()
        baseline_tokens  = self._estimate_baseline_tokens()
        self.total_ccm_tokens      += ccm_tokens
        self.total_baseline_tokens += baseline_tokens

        print(
            f"[CCM] Context ready: {ccm_tokens} tokens "
            f"(baseline would be ~{baseline_tokens})"
        )
        return context

    def process_tool_result(
        self,
        tool_name: str,
        raw_result: dict,
        query_used: str = "",
    ) -> str:
        """
        Compress a tool result and store it in SemanticMemory.

        Call this immediately after executing any tool.
        Use the returned string (not the raw result) in your LLM prompt.
        """
        print(f"[CCM] Compressing tool result: {tool_name}")

        constraints = [f["value"] for f in self.working_memory.get_critical_facts()]

        try:
            compressed = self.compressor.compress(
                tool_result=raw_result,
                tool_name=tool_name,
                user_constraints=constraints,
            )
        except Exception as exc:
            print(f"[CCM] Compressor error: {exc}")
            compressed = str(raw_result)[:300]

        try:
            self.semantic_memory.add(
                compressed_result=compressed,
                tool_name=tool_name,
                query_used=query_used or tool_name,
                turn_number=self.turn_count,
            )
        except Exception as exc:
            print(f"[CCM] SemanticMemory.add error (non-fatal): {exc}")

        return compressed

    def process_agent_response(
        self,
        user_message: str,
        agent_response: str,
        tool_calls_made: list = None,
    ):
        """
        Update memory after the agent responds.

        Call this AFTER every agent response.
        """
        self.conversation_history.append({"role": "user",      "content": user_message})
        self.conversation_history.append({"role": "assistant",  "content": agent_response})

        self.turn_buffer.append(
            f"User: {user_message}\nAssistant: {agent_response}"
        )

        if len(self.turn_buffer) >= EPISODE_EVERY_N_TURNS:
            self._create_episode_summary()
            self.turn_buffer = []

    # ── Internal helpers ─────────────────────────────────────────

    def _create_episode_summary(self):
        """Summarise the current turn buffer and store in EpisodicMemory."""
        if not self.turn_buffer:
            return

        from groq import Groq
        from travel_agent.prompts import EPISODIC_SUMMARY_PROMPT

        print(f"[CCM] Creating episodic summary ({len(self.turn_buffer)} turns)…")

        wm_snapshot = self.working_memory.format_for_prompt()
        turns_text  = "\n\n---\n\n".join(self.turn_buffer)

        prompt = EPISODIC_SUMMARY_PROMPT.format(
            turns=turns_text,
            working_memory_snapshot=wm_snapshot,
        )

        try:
            client   = Groq(api_key=os.getenv("GROQ_API_KEY"))
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Create a concise 2-3 sentence episodic memory summary. "
                            "Be specific: include exact names, prices, decisions. "
                            "Do NOT duplicate facts already in working memory."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=150,
                temperature=0.0,
            )
            summary = response.choices[0].message.content.strip()

            start_turn = self.turn_count - len(self.turn_buffer)
            self.episodic_memory.add(
                summary_text=summary,
                turn_range=(start_turn, self.turn_count),
            )
            print(f"[CCM] Episode stored: {summary[:80]}…")

        except Exception as exc:
            print(f"[CCM] Episode creation failed (non-fatal): {exc}")

    def _estimate_baseline_tokens(self) -> int:
        """Estimate tokens the baseline agent would use (full raw history)."""
        return sum(
            _count_tokens(str(t.get("content", "")))
            for t in self.conversation_history
        )

    # ── UI / metrics ─────────────────────────────────────────────

    def get_memory_state(self) -> dict:
        """Full state snapshot for UI display and debugging."""
        ep_count  = self.episodic_memory.get_count()
        sem_count = self.semantic_memory.get_count()
        comp_stats = self.compressor.get_compression_stats()

        baseline = self.total_baseline_tokens
        ccm      = self.total_ccm_tokens
        ratio    = round(baseline / max(ccm, 1), 2)

        return {
            "working_memory":     self.working_memory.format_for_prompt(),
            "working_memory_raw": self.working_memory.get_all(),
            "episodic_count":     ep_count,
            "semantic_count":     sem_count,
            "episodic_entries":   self.episodic_memory.get_all_active(),
            "semantic_entries":   self.semantic_memory.get_all_active(),
            "turn_count":         self.turn_count,
            "token_metrics": {
                "baseline_tokens_used": baseline,
                "ccm_tokens_used":      ccm,
                "compression_ratio":    ratio,
                "tokens_saved":         baseline - ccm,
            },
            "compression_stats":    comp_stats,
            "assembler_breakdown":  self.assembler.get_breakdown(),
        }

    def reset(self):
        """Full reset for a new conversation. Clears ALL memory tiers."""
        print("[CCM] Resetting all memory…")
        self.working_memory.reset()
        self.episodic_memory.reset()
        self.semantic_memory.reset()
        self.compressor.reset_stats()
        self.conversation_history  = []
        self.turn_buffer           = []
        self.turn_count            = 0
        self.total_baseline_tokens = 0
        self.total_ccm_tokens      = 0
        print("[CCM] Reset complete")