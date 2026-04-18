# ccm/stale_detector.py
#
# Detects when the user overrides, cancels, or replaces something
# they said earlier, then removes all traces from memory.
#
# WHY THIS IS NEEDED:
#   Without this, if the user says "Plan a Bali trip" then later
#   "Scratch Bali, do Switzerland instead", the agent will keep
#   injecting Bali hotels, Bali weather, and Bali flights into
#   every subsequent prompt. The conversation becomes incoherent.
#
# HOW IT WORKS:
#   1. Fast keyword pre-filter (no LLM call yet)
#      If "scratch", "cancel", "forget", etc. are NOT in the message
#      → return immediately (most messages pass through without LLM cost)
#
#   2. If a signal word IS found, ask the LLM:
#      "Given the current memory and this new message,
#       which memory keys are being overridden?"
#
#   3. For each overridden key:
#      a. Get the old value from WorkingMemory
#      b. Remove the key from WorkingMemory
#      c. Add old value to cancelled list
#      d. Call episodic.mark_stale_by_content(old_value)
#      e. Call semantic.mark_stale_by_content(old_value)
#
# AGENT-CENTRIC:
#   Operates on memory keys and values — not domain concepts.
#   Works for travel, medical, legal, or any other agent.

import os
import json
from groq import Groq
from dotenv import load_dotenv
from travel_agent.prompts import STALE_DETECTION_PROMPT

load_dotenv()

# Fast pre-filter: words that signal the user is overriding something.
# Only language patterns, not domain-specific terms.
OVERRIDE_SIGNALS = [
    "scratch", "forget", "cancel", "instead", "actually",
    "change", "never mind", "nevermind", "drop", "not anymore",
    "changed my mind", "no longer", "skip", "disregard",
    "replace", "switch", "swap", "rather", "different",
    "actually no", "forget about", "let's not", "lets not",
    "ignore", "remove", "scrap", "ditch", "new plan",
]


class StaleDetector:
    """
    Detects and removes stale/overridden context from ALL memory tiers.

    Public API:
      check_and_clean(user_message, working_memory,
                      episodic_memory=None,
                      semantic_memory=None)  → dict
    """

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model  = "llama-3.3-70b-versatile"

    def _has_override_signal(self, message: str) -> bool:
        """Fast keyword check — avoids LLM call for normal messages."""
        msg = message.lower()
        return any(sig in msg for sig in OVERRIDE_SIGNALS)

    def check_and_clean(
        self,
        user_message: str,
        working_memory,
        episodic_memory=None,
        semantic_memory=None,
    ) -> dict:
        """
        Detect overrides and clean ALL memory tiers.

        Parameters
        ----------
        user_message    : str
        working_memory  : WorkingMemory instance
        episodic_memory : EpisodicMemory | None
        semantic_memory : SemanticMemory | None

        Returns
        -------
        dict
            {has_override, overridden_keys, cancelled_values, reason}
        """
        _no_override = {
            "has_override":     False,
            "overridden_keys":  [],
            "cancelled_values": [],
            "reason":           "",
        }

        if not self._has_override_signal(user_message):
            return _no_override

        print("[StaleDetector] Override signal detected — consulting LLM…")

        # Build flat memory dict for the LLM
        current_memory = working_memory.get_all()
        flat: dict = {}
        for priority in ("critical", "important", "contextual"):
            for fact in current_memory["facts"][priority]:
                flat[fact["key"]] = fact["value"]
        flat["_decisions"] = current_memory.get("decisions", [])
        flat["_cancelled"] = current_memory.get("cancelled", [])

        if not flat or flat == {"_decisions": [], "_cancelled": []}:
            print("[StaleDetector] Memory empty — nothing to override")
            return _no_override

        prompt = STALE_DETECTION_PROMPT.format(
            message=user_message,
            current_memory=json.dumps(flat, indent=2),
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You detect when a user overrides a previous statement. "
                            "Return ONLY valid JSON with these exact keys: "
                            "has_override (bool), overridden_keys (list of strings), "
                            "cancelled_values (list of strings), reason (string). "
                            "Be conservative — only flag explicit overrides."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
                temperature=0.0,
            )

            raw = response.choices[0].message.content.strip()

            # Strip markdown fences if present
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()

            start, end = raw.find("{"), raw.rfind("}") + 1
            if start >= 0 and end > start:
                raw = raw[start:end]

            result = json.loads(raw)

            # Validate structure
            if not isinstance(result, dict):
                raise ValueError("Response is not a dict")
            result.setdefault("has_override",     False)
            result.setdefault("overridden_keys",  [])
            result.setdefault("cancelled_values", [])
            result.setdefault("reason",           "")

        except (json.JSONDecodeError, ValueError, Exception) as exc:
            print(f"[StaleDetector] LLM parse error: {exc}")
            return _no_override

        if not result.get("has_override"):
            print("[StaleDetector] No override found")
            return result

        overridden      = result.get("overridden_keys", [])
        cancelled_vals  = result.get("cancelled_values", [])

        print(f"[StaleDetector] Overriding keys: {overridden}")
        print(f"[StaleDetector] Reason: {result.get('reason', '')}")

        for key in overridden:
            old_value = working_memory.get(key)
            working_memory.remove_by_key(key)

            if old_value and isinstance(old_value, str):
                working_memory.add_cancelled(old_value)

                if episodic_memory:
                    count = episodic_memory.mark_stale_by_content(old_value)
                    print(f"[StaleDetector] Episodic: {count} entries marked stale")

                if semantic_memory:
                    count = semantic_memory.mark_stale_by_content(old_value)
                    print(f"[StaleDetector] Semantic: {count} entries marked stale")

        # Also mark stale using the cancelled_values list directly
        for val in cancelled_vals:
            if val and len(val) > 3:
                if episodic_memory:
                    episodic_memory.mark_stale_by_content(val)
                if semantic_memory:
                    semantic_memory.mark_stale_by_content(val)

        return result