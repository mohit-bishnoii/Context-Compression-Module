# ccm/extractor.py
#
# Extracts durable facts from user messages and stores them
# in WorkingMemory (Tier 1).
#
# WHAT COUNTS AS A FACT:
#   ✓ Constraints:  "I'm allergic to shellfish" → allergy_shellfish: critical
#   ✓ Preferences:  "max 2 activities per day"  → activity_limit: important
#   ✓ Decisions:    "budget is $3000 max"        → budget_maximum: critical
#   ✗ NOT questions:  "find hotels in Shinjuku" (search query, not a user fact)
#   ✗ NOT hypotheticals: "what if we did Paris?"
#
# PRIORITY LOGIC:
#   critical   → hard constraint; ignoring it causes harm/major mistake
#   important  → strong preference; ignoring it noticeably degrades response
#   contextual → background info; useful but not decision-critical
#
# AGENT-CENTRIC: works for travel, medical, legal, or any agent.

import os
import json
from groq import Groq
from dotenv import load_dotenv
from travel_agent.prompts import EXTRACTION_PROMPT

load_dotenv()

# ── Medical/safety keywords that MUST be critical ───────────────
# If the LLM under-classifies a medical constraint, we promote it.
_CRITICAL_TRIGGERS = [
    "allerg", "anaphylax", "medical", "severe", "cannot eat",
    "must not", "never eat", "life-threatening", "epipen",
    "intoleran", "celiac", "diabetic", "disabled", "wheelchair",
]


class MemoryExtractor:
    """
    Domain-agnostic fact extractor.

    Main entry point: extract_and_update(user_message, working_memory)
    Calls extract() internally, validates results, promotes priority
    for safety-critical facts, then stores in WorkingMemory.
    """

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model  = "llama-3.1-8b-instant"

    def extract(self, user_message: str, current_memory: dict) -> list:
        """
        Extract facts from a single user message.

        Returns
        -------
        list[dict]
            Each dict: {key, value, category, priority}
            Empty list if nothing worth storing.
        """
        if len(user_message.strip()) < 10:
            return []

        prompt = EXTRACTION_PROMPT.format(
            message=user_message,
            current_memory=json.dumps(
                current_memory.get("facts", {}), indent=2
            ),
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You extract facts and return JSON only. "
                            "Return ONLY a valid JSON object with a 'facts' array. "
                            "No markdown, no explanation, no text outside JSON. "
                            "IMPORTANT: Medical allergies (shellfish, nuts, gluten, etc.) "
                            "and hard budget limits ('maximum', 'cannot exceed') "
                            "MUST be classified as priority='critical'."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=600,
                temperature=0.0,
            )

            raw = response.choices[0].message.content.strip()

            # Strip markdown
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()

            # Find JSON object
            start, end = raw.find("{"), raw.rfind("}") + 1
            if start >= 0 and end > start:
                raw = raw[start:end]

            parsed = json.loads(raw)
            facts  = parsed.get("facts", [])

        except json.JSONDecodeError as exc:
            print(f"[Extractor] JSON parse failed: {exc}")
            return []
        except Exception as exc:
            print(f"[Extractor] Error: {exc}")
            return []

        # ── Validate + safety-promote ────────────────────────────
        valid = []
        for f in facts:
            if not isinstance(f, dict):
                continue
            key      = (f.get("key")      or "").strip()
            value    = (f.get("value")    or "").strip()
            priority = (f.get("priority") or "contextual")
            category = (f.get("category") or "information")

            if not key or not value:
                print(f"[Extractor] Skipped incomplete fact: {f}")
                continue
            if priority not in ("critical", "important", "contextual"):
                priority = "contextual"

            # Safety promotion: if value mentions medical/safety keywords
            # and LLM didn't classify as critical, promote it
            val_lower = value.lower()
            if priority != "critical" and any(
                kw in val_lower for kw in _CRITICAL_TRIGGERS
            ):
                print(
                    f"[Extractor] Safety-promoting to critical: {key} = {value}"
                )
                priority = "critical"

            valid.append({
                "key":      key,
                "value":    value,
                "category": category,
                "priority": priority,
            })

        if valid:
            print(f"[Extractor] Found {len(valid)} facts:")
            for f in valid:
                print(f"  [{f['priority']:11}] {f['key']:30} → {f['value']}")
        else:
            print("[Extractor] No new facts in this message")

        return valid

    def extract_and_update(
        self, user_message: str, working_memory
    ) -> list:
        """
        Extract facts and store them in WorkingMemory.

        This is the method called by CCMCore every turn.
        Returns the list of extracted facts (for logging).
        """
        current_state = working_memory.get_all()
        facts = self.extract(user_message, current_state)
        if facts:
            working_memory.add_facts(facts)
        return facts