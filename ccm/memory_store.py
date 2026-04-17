# ccm/memory_store.py
#
# Tier 1: Working Memory
#
# WHAT IT IS:
#   A single JSON file on disk that stores facts extracted from
#   the conversation. It is ALWAYS fully injected into every
#   LLM prompt. This is what makes critical constraints (like
#   a shellfish allergy) impossible to forget.
#
# THREE PRIORITY LEVELS:
#   critical   — always injected first, hard constraints
#   important  — always injected if budget allows
#   contextual — stored but retrieved via RAG when relevant
#
# LIFECYCLE:
#   1. User says "I'm allergic to shellfish"
#   2. Extractor marks it as {key:"allergy_shellfish", priority:"critical"}
#   3. memory.add_facts([...]) stores it here
#   4. Every subsequent LLM call includes:
#        [CRITICAL CONSTRAINTS]
#        • severely allergic to shellfish
#   5. Agent always sees it regardless of conversation length
#
# AGENT-CENTRIC:
#   This class knows nothing about travel, allergies, or budgets.
#   It only knows about priority levels and key-value pairs.

import os
import json
import tiktoken
from datetime import datetime
from typing import Any

os.makedirs("data", exist_ok=True)   # ensure dir exists at import time

MEMORY_FILE_PATH         = "data/working_memory.json"
MAX_WORKING_MEMORY_TOKENS = 400       # hard cap for prompt injection


def _default_memory() -> dict:
    return {
        "facts": {
            "critical":   [],
            "important":  [],
            "contextual": [],
        },
        "decisions":  [],
        "cancelled":  [],
        "turn_count": 0,
        "conversation_id": "",
        "last_updated": "",
    }


def _count_tokens(text: str) -> int:
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


class WorkingMemory:
    """
    Tier 1 — Working Memory.

    Persists to ``data/working_memory.json``.
    Survives program restarts (important for multi-session use).

    Public API:
      add_facts(facts: list)            — store extracted facts
      remove_by_key(key: str)           — delete one fact (stale detector)
      remove_by_value_substring(s: str) — delete facts mentioning s
      add_cancelled(item: str)          — record a cancelled value
      add_decision(decision: str)       — record a confirmed decision
      get(key: str) → Any               — retrieve a fact value by key
      get_all() → dict                  — full snapshot
      get_critical_facts() → list       — list of critical fact dicts
      get_important_facts() → list
      format_for_prompt() → str         — formatted string for injection
      increment_turn()                  — bump turn counter
      reset()                           — wipe everything
    """

    def __init__(self):
        os.makedirs("data", exist_ok=True)
        self.memory = self._load()

    # ── Persistence ─────────────────────────────────────────────

    def _load(self) -> dict:
        if os.path.exists(MEMORY_FILE_PATH):
            try:
                with open(MEMORY_FILE_PATH, "r") as f:
                    loaded = json.load(f)
                if "facts" not in loaded:
                    print("[WorkingMemory] Old format — migrating")
                    return _default_memory()
                print("[WorkingMemory] Loaded from disk")
                return loaded
            except Exception as exc:
                print(f"[WorkingMemory] Load error ({exc}) — fresh start")
        return _default_memory()

    def _save(self):
        self.memory["last_updated"] = datetime.now().isoformat()
        try:
            with open(MEMORY_FILE_PATH, "w") as f:
                json.dump(self.memory, f, indent=2)
        except Exception as exc:
            print(f"[WorkingMemory] Save error: {exc}")

    # ── Write ───────────────────────────────────────────────────

    def add_facts(self, facts: list):
        """
        Add a list of extracted facts.

        Each fact dict must have: key, value, category, priority.
        Handles deduplication: if the key already exists, update
        the value if it changed.
        """
        if not facts:
            return

        changed = False
        for fact in facts:
            key      = (fact.get("key")      or "").strip()
            value    = (fact.get("value")    or "").strip()
            priority = (fact.get("priority") or "contextual")
            category = (fact.get("category") or "information")

            if not key or not value:
                continue
            if priority not in ("critical", "important", "contextual"):
                priority = "contextual"

            # Search all buckets for existing key
            found = False
            for p in ("critical", "important", "contextual"):
                for i, existing in enumerate(self.memory["facts"][p]):
                    if existing.get("key") == key:
                        if existing.get("value") != value:
                            print(
                                f"[WorkingMemory] Update [{p}] {key}: "
                                f"'{existing['value']}' → '{value}'"
                            )
                            self.memory["facts"][p][i]["value"]    = value
                            self.memory["facts"][p][i]["category"] = category
                            changed = True
                        found = True
                        break
                if found:
                    break

            if not found:
                new_fact = {
                    "key":          key,
                    "value":        value,
                    "category":     category,
                    "priority":     priority,
                    "added_at_turn": self.memory["turn_count"],
                }
                self.memory["facts"][priority].append(new_fact)
                print(f"[WorkingMemory] New [{priority:11}] {key}: {value}")
                changed = True

        if changed:
            self._save()

    def remove_by_key(self, key: str):
        """Remove a fact by its key. Used by StaleDetector."""
        removed = False
        for priority in ("critical", "important", "contextual"):
            before = len(self.memory["facts"][priority])
            self.memory["facts"][priority] = [
                f for f in self.memory["facts"][priority]
                if f.get("key") != key
            ]
            if len(self.memory["facts"][priority]) < before:
                print(f"[WorkingMemory] Removed fact: {key}")
                removed = True
        if removed:
            self._save()

    def remove_by_value_substring(self, substring: str) -> list:
        """
        Remove all facts whose value contains ``substring``.
        Used by StaleDetector for broader cancellations.
        Returns list of removed keys.
        """
        removed_keys = []
        for priority in ("critical", "important", "contextual"):
            to_remove = [
                f for f in self.memory["facts"][priority]
                if substring.lower() in f.get("value", "").lower()
            ]
            for f in to_remove:
                removed_keys.append(f["key"])
            self.memory["facts"][priority] = [
                f for f in self.memory["facts"][priority]
                if substring.lower() not in f.get("value", "").lower()
            ]
        if removed_keys:
            print(
                f"[WorkingMemory] Removed facts mentioning "
                f"'{substring}': {removed_keys}"
            )
            self._save()
        return removed_keys

    def add_cancelled(self, item: str):
        """Record something as explicitly cancelled (shown to agent)."""
        if item and item not in self.memory["cancelled"]:
            self.memory["cancelled"].append(item)
            self._save()
            print(f"[WorkingMemory] Cancelled: {item}")

    def add_decision(self, decision: str):
        """Record a confirmed decision."""
        if decision and decision not in self.memory["decisions"]:
            self.memory["decisions"].append(decision)
            self._save()

    # ── Read ───────────────────────────────────────────────────

    def get(self, key: str, default=None) -> Any:
        """Get a fact value by key, searching all priority levels."""
        for priority in ("critical", "important", "contextual"):
            for fact in self.memory["facts"][priority]:
                if fact.get("key") == key:
                    return fact.get("value", default)
        return default

    def get_all(self) -> dict:
        """Full memory snapshot (a copy)."""
        return self.memory.copy()

    def get_critical_facts(self) -> list:
        return self.memory["facts"]["critical"]

    def get_important_facts(self) -> list:
        return self.memory["facts"]["important"]

    def get_all_facts_as_text_list(self) -> list:
        """All fact values as plain strings. Used for embedding."""
        texts = []
        for p in ("critical", "important", "contextual"):
            for f in self.memory["facts"][p]:
                if f.get("value"):
                    texts.append(f["value"])
        return texts

    def increment_turn(self):
        self.memory["turn_count"] += 1
        self._save()

    # ── Prompt formatting ────────────────────────────────────────

    def format_for_prompt(self) -> str:
        """
        Build the string that gets injected into every LLM prompt.

        Layout:
          [CRITICAL CONSTRAINTS]
            • <value>
          [USER PREFERENCES]
            • <value>
          [DECISIONS MADE]
            • <decision>
          [CANCELLED — DO NOT REVISIT]
            • <item>

        Enforces MAX_WORKING_MEMORY_TOKENS hard cap.
        Critical facts are NEVER truncated.
        """
        lines = []
        total_tokens = 0

        def _add_line(line: str) -> bool:
            nonlocal total_tokens
            t = _count_tokens(line)
            if total_tokens + t > MAX_WORKING_MEMORY_TOKENS:
                return False
            lines.append(line)
            total_tokens += t
            return True

        # Critical — must always appear
        critical = self.memory["facts"]["critical"]
        if critical:
            _add_line("[CRITICAL CONSTRAINTS]")
            for fact in critical:
                _add_line(f"  • {fact['value']}")

        # Important
        important = self.memory["facts"]["important"]
        if important:
            if _add_line("[USER PREFERENCES]"):
                for fact in important:
                    if not _add_line(f"  • {fact['value']}"):
                        break

        # Decisions
        decisions = self.memory["decisions"]
        if decisions:
            if _add_line("[DECISIONS MADE]"):
                for d in decisions[-5:]:
                    if not _add_line(f"  • {d}"):
                        break

        # Cancelled
        cancelled = self.memory["cancelled"]
        if cancelled:
            if _add_line("[CANCELLED — DO NOT REVISIT]"):
                for c in cancelled:
                    if not _add_line(f"  • {c}"):
                        break

        if not lines:
            return "[NO USER PREFERENCES CAPTURED YET]"

        return "\n".join(lines)

    def reset(self):
        """Full reset for a new conversation."""
        self.memory = _default_memory()
        self._save()
        print("[WorkingMemory] Reset complete")