# ccm/ccm_core.py
#
# The main orchestrator of the Context Compression Module.
# This is the single entry point for any agent using the CCM.
#
# USAGE (agent-centric, plug-and-play):
#
#   from ccm.ccm_core import ContextCompressionModule
#
#   ccm = ContextCompressionModule()
#
#   # Before sending to LLM:
#   context = ccm.process_user_message(user_message)
#
#   # After getting tool result:
#   compressed = ccm.process_tool_result(tool_name, raw_result)
#
#   # After agent responds:
#   ccm.process_agent_response(user_msg, agent_response)
#
#   # Get memory state for UI display:
#   state = ccm.get_memory_state()
#
# That is the entire interface. The agent does not need to know
# anything about memory tiers, ChromaDB, or embeddings.

import os
# Silence ChromaDB telemetry warnings
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

import json
import tiktoken
from dotenv import load_dotenv

from ccm.memory_store import WorkingMemory
from ccm.episodic_memory import EpisodicMemory
from ccm.semantic_memory import SemanticMemory
from ccm.extractor import MemoryExtractor
from ccm.compressor import ToolCompressor
from ccm.stale_detector import StaleDetector
from ccm.retriever import Retriever
from ccm.assembler import ContextAssembler

load_dotenv()

# How many turns before creating an episodic summary
EPISODE_EVERY_N_TURNS = 4


def count_tokens(text: str) -> int:
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4


class ContextCompressionModule:
    """
    The Context Compression Module — plug-and-play middleware.

    Drop this between any user interface and any LLM agent.
    The CCM handles all memory management automatically.

    WHAT IT DOES EACH TURN:
      1. Extracts facts from user message → Working Memory
      2. Checks for stale/cancelled context → cleans memory
      3. Retrieves relevant memories via RAG
      4. Assembles compressed context packet
      5. Returns context for agent to use

      After tool calls:
      6. Compresses tool results
      7. Stores in Semantic Memory

      After agent responds:
      8. Periodically creates Episodic summary
      9. Updates decisions in Working Memory

    AGENT-CENTRIC:
      The agent calls three methods: process_user_message,
      process_tool_result, process_agent_response.
      Everything else is internal to the CCM.
    """

    def __init__(self, use_reranking: bool = True):
        """
        Initialize all CCM components.

        Parameters:
          use_reranking: If True, use LLM re-ranking in retrieval.
                        Slower but more precise. Default True.
                        Set False for faster demo if needed.
        """
        print("[CCM] Initializing Context Compression Module...")

        # Three memory tiers
        self.working_memory = WorkingMemory()
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()

        # Processing components
        self.extractor = MemoryExtractor()
        self.compressor = ToolCompressor()
        self.stale_detector = StaleDetector()

        # Retrieval and assembly
        self.retriever = Retriever(
            episodic_memory=self.episodic_memory,
            semantic_memory=self.semantic_memory,
            use_reranking=use_reranking
        )
        self.assembler = ContextAssembler()

        # Conversation state
        self.conversation_history = []  # Full raw history
        self.turn_buffer = []           # Turns since last episode
        self.turn_count = 0
        self.total_baseline_tokens = 0  # Track what baseline would use
        self.total_ccm_tokens = 0       # Track what we actually use

        print("[CCM] Ready")

    def process_user_message(self, user_message: str) -> str:
        """
        Process a user message and return compressed context.

        This is the main method called before every LLM call.

        Flow:
          1. Extract facts → update Working Memory
          2. Detect stale context → clean memory
          3. Retrieve relevant memories via RAG
          4. Assemble compressed context packet
          5. Return context string

        Parameters:
          user_message: The raw user input

        Returns:
          Compressed context string to prepend to the LLM prompt
        """
        self.turn_count += 1
        self.working_memory.increment_turn()

        print(f"\n{'='*50}")
        print(f"[CCM] Processing turn {self.turn_count}")
        print(f"{'='*50}")

        # Step 1: Extract facts from user message
        print("[CCM] Step 1: Extracting facts...")
        self.extractor.extract_and_update(
            user_message,
            self.working_memory
        )

        # Step 2: Check for stale context
        print("[CCM] Step 2: Checking for stale context...")
        stale_result = self.stale_detector.check_and_clean(
            user_message,
            self.working_memory,
            self.episodic_memory,
            self.semantic_memory         
        )
        if stale_result.get("has_override"):
            print(
                f"[CCM] Stale context cleaned: "
                f"{stale_result.get('cancelled_values', [])}"
            )

        # Step 3: Retrieve relevant memories
        print("[CCM] Step 3: Retrieving relevant memories...")
        retrieved = self.retriever.retrieve(
            query=user_message,
            n_episodic=4,
            n_semantic=3
        )

        # Step 4: Assemble context packet
        print("[CCM] Step 4: Assembling context...")
        context = self.assembler.assemble(
            working_memory=self.working_memory,
            retrieved=retrieved,
            recent_turns=self.conversation_history,
            max_recent_turns=3
        )

        # Track token metrics
        ccm_tokens = self.assembler.get_last_token_count()
        self.total_ccm_tokens += ccm_tokens

        # Calculate what baseline would use
        baseline_tokens = self._estimate_baseline_tokens()
        self.total_baseline_tokens += baseline_tokens

        print(
            f"[CCM] Context ready: {ccm_tokens} tokens "
            f"(baseline would use ~{baseline_tokens} tokens)"
        )

        return context

    def process_tool_result(
        self,
        tool_name: str,
        raw_result: dict,
        query_used: str = ""
    ) -> str:
        """
        Compress a tool result and store it in Semantic Memory.

        Called immediately after every tool execution.

        Parameters:
          tool_name:  Name of the tool that was called
          raw_result: Raw dict returned by the tool
          query_used: The query that was passed to the tool

        Returns:
          Compressed result string (use this instead of raw result)
        """
        print(f"[CCM] Compressing tool result: {tool_name}")

        # Get current constraints for conflict checking
        constraints = [
            f["value"]
            for f in self.working_memory.get_critical_facts()
        ]

        # Compress the tool result
        compressed = self.compressor.compress(
            tool_result=raw_result,
            tool_name=tool_name,
            user_constraints=constraints
        )

        # Store compressed result in Semantic Memory
        self.semantic_memory.add(
            compressed_result=compressed,
            tool_name=tool_name,
            query_used=query_used or tool_name,
            turn_number=self.turn_count
        )

        return compressed

    def process_agent_response(
        self,
        user_message: str,
        agent_response: str,
        tool_calls_made: list = None
    ):
        """
        Update memory after agent responds.

        Called after every agent response.

        Flow:
          1. Add user message and response to history
          2. Add turns to episode buffer
          3. If buffer full, create episodic summary
          4. Extract any decisions from agent response

        Parameters:
          user_message:    The user's message this turn
          agent_response:  The agent's response
          tool_calls_made: List of tool calls made this turn
        """
        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": agent_response
        })

        # Add to episode buffer
        self.turn_buffer.append(
            f"User: {user_message}\nAssistant: {agent_response}"
        )

        # Create episodic summary every N turns
        if len(self.turn_buffer) >= EPISODE_EVERY_N_TURNS:
            self._create_episode_summary()
            self.turn_buffer = []

    def _create_episode_summary(self):
        """
        Summarize the current episode buffer and store in Episodic Memory.
        Called automatically every N turns.
        """
        if not self.turn_buffer:
            return

        from groq import Groq
        from travel_agent.prompts import EPISODIC_SUMMARY_PROMPT

        print(
            f"[CCM] Creating episodic summary "
            f"({len(self.turn_buffer)} turns)..."
        )

        # Get working memory snapshot to avoid duplicating in summary
        wm_snapshot = self.working_memory.format_for_prompt()

        turns_text = "\n\n---\n\n".join(self.turn_buffer)

        prompt = EPISODIC_SUMMARY_PROMPT.format(
            turns=turns_text,
            working_memory_snapshot=wm_snapshot
        )

        try:
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You create concise memory summaries. "
                            "Be specific. Use exact numbers and names."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=150,
                temperature=0.0
            )

            summary = response.choices[0].message.content.strip()

            # Store in episodic memory
            start_turn = self.turn_count - len(self.turn_buffer)
            end_turn = self.turn_count

            self.episodic_memory.add(
                summary_text=summary,
                turn_range=(start_turn, end_turn)
            )

            print(f"[CCM] Episode stored: {summary[:80]}...")

        except Exception as e:
            print(f"[CCM] Episode creation failed: {e}")

    def _estimate_baseline_tokens(self) -> int:
        """
        Estimate how many tokens the baseline agent would use.
        Counts the full raw conversation history.
        Used to calculate compression ratio for metrics.
        """
        total = 0
        for turn in self.conversation_history:
            total += count_tokens(str(turn.get("content", "")))
        return total

    def get_memory_state(self) -> dict:
        """
        Get complete memory state for UI display and debugging.

        Returns dict with all tier contents and metrics.
        Used by Gradio to populate the memory state panel.
        """
        ep_count = self.episodic_memory.get_count()
        sem_count = self.semantic_memory.get_count()

        compression_stats = self.compressor.get_compression_stats()

        baseline = self.total_baseline_tokens
        ccm = self.total_ccm_tokens
        ratio = round(baseline / max(ccm, 1), 2)

        return {
            "working_memory": self.working_memory.format_for_prompt(),
            "working_memory_raw": self.working_memory.get_all(),
            "episodic_count": ep_count,
            "semantic_count": sem_count,
            "episodic_entries": self.episodic_memory.get_all_active(),
            "turn_count": self.turn_count,
            "token_metrics": {
                "baseline_tokens_used": baseline,
                "ccm_tokens_used": ccm,
                "compression_ratio": ratio,
                "tokens_saved": baseline - ccm
            },
            "compression_stats": compression_stats,
            "assembler_breakdown": self.assembler.get_breakdown()
        }

    def reset(self):
        """
        Full reset for a new conversation.
        Clears all memory tiers and resets counters.
        """
        print("[CCM] Resetting all memory...")
        self.working_memory.reset()
        self.episodic_memory.reset()
        self.semantic_memory.reset()
        self.compressor.reset_stats()
        self.conversation_history = []
        self.turn_buffer = []
        self.turn_count = 0
        self.total_baseline_tokens = 0
        self.total_ccm_tokens = 0
        print("[CCM] Reset complete")