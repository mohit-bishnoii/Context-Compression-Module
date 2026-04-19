# ccm/compressor.py
#
# Compresses raw tool results (600–800 tokens) into brief summaries
# (~60–100 tokens) before they are stored in SemanticMemory or
# returned to the agent.
#
# WHY THIS MATTERS:
#   Without compression:  15 tool calls × 600 tokens = 9,000 tokens
#   With compression:     15 tool calls × 80 tokens  = 1,200 tokens
#   Saving: ~7,800 tokens per conversation
#
# HOW IT WORKS:
#   1. Receive raw dict from tool (e.g. places_search result)
#   2. JSON-serialise it
#   3. LLM call: "Compress this. Max 80 words. Flag constraint conflicts."
#   4. Return compressed string
#
# CONSTRAINT CONFLICT CHECK:
#   The critical constraints from WorkingMemory are passed in.
#   If a tool result conflicts (e.g. shellfish restaurant for someone
#   with a shellfish allergy), the compression includes a ⚠️ flag.
#
# AGENT-CENTRIC: knows no travel concepts. Works with any tool.

import os
import json
from groq import Groq
from dotenv import load_dotenv
from travel_agent.prompts import COMPRESSION_PROMPT

load_dotenv()


class ToolCompressor:
    """
    Compresses raw tool results into compact summaries.

    Public API:
      compress(tool_result, tool_name, user_constraints) → str
      get_compression_stats()                             → dict
      reset_stats()
    """

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model  = "llama-3.1-8b-instant"
        self.stats  = {
            "total_calls":          0,
            "total_tokens_before":  0,
            "total_tokens_after":   0,
        }

    def compress(
        self,
        tool_result: dict,
        tool_name: str,
        user_constraints: list = None,
    ) -> str:
        """
        Compress a tool result dict into a short summary string.

        Parameters
        ----------
        tool_result      : dict   Raw output from the tool function
        tool_name        : str    Name of the tool
        user_constraints : list   Critical constraint strings for conflict check

        Returns
        -------
        str   Compressed summary (~60–100 words)
        """
        if not tool_result:
            return f"{tool_name}: no results returned."

        tool_result_str = json.dumps(tool_result, indent=2)
        tokens_before   = len(tool_result_str) // 4
        self.stats["total_tokens_before"] += tokens_before

        constraints_text = "None specified"
        if user_constraints:
            constraints_text = "\n".join(f"  - {c}" for c in user_constraints)

        prompt = COMPRESSION_PROMPT.format(
            tool_type=tool_name,
            user_constraints=constraints_text,
            tool_result=tool_result_str,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You compress tool results into brief plain-text summaries. "
                            "No markdown. No bold. No asterisks. No headers. "
                            "Use plain sentences or simple dashes for lists only. "
                            "Preserve all numbers exactly. Maximum 150 words. "
                            "Flag any constraint conflicts with ⚠️."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=375,
                temperature=0.0,
            )

            compressed  = response.choices[0].message.content.strip()
            tokens_after = len(compressed) // 4
            self.stats["total_tokens_after"] += tokens_after
            self.stats["total_calls"]        += 1

            ratio = tokens_before / max(tokens_after, 1)
            print(
                f"[Compressor] {tool_name}: "
                f"{tokens_before}→{tokens_after} tokens ({ratio:.1f}x)"
            )
            return compressed

        except Exception as exc:
            print(f"[Compressor] Error: {exc} — using fallback")
            return self._fallback_compress(tool_result, tool_name)

    def _fallback_compress(self, result: dict, tool_name: str) -> str:
        """Rule-based fallback compression when LLM call fails."""
        try:
            if tool_name == "places_search":
                location = result.get("location", "")
                all_r    = result.get("all_results", result.get("results", []))
                lines    = [f"{tool_name} in {location}:"]
                for item in all_r[:3]:
                    name  = item.get("name", "?")
                    price = item.get("price_per_night", item.get("price_range", "?"))
                    rating = item.get("rating", "")
                    lines.append(f"  {name}: ${price} ★{rating}")
                return "\n".join(lines)

            elif tool_name == "web_search":
                route    = result.get("route", "")
                cheapest = result.get("cheapest_price", "?")
                return f"Flights {route}: cheapest ${cheapest}"

            elif tool_name == "weather_fetch":
                city  = result.get("city", "")
                cond  = result.get("current_conditions", {})
                temp  = cond.get("temperature_f", "?")
                desc  = cond.get("description", "")
                return f"Weather {city}: {temp}°F, {desc}"

            elif tool_name == "budget_tracker":
                remaining = result.get("remaining", "?")
                total     = result.get("total_budget", "?")
                spent     = result.get("total_spent", "?")
                return f"Budget: ${spent} spent of ${total}. ${remaining} remaining."

            else:
                raw = json.dumps(result)
                return (raw[:200] + "…") if len(raw) > 200 else raw

        except Exception:
            return f"{tool_name} result (compression failed)"

    def get_compression_stats(self) -> dict:
        before = self.stats["total_tokens_before"]
        after  = self.stats["total_tokens_after"]
        calls  = self.stats["total_calls"]
        ratio  = before / max(after, 1)
        saved  = before - after
        return {
            "total_tool_calls_compressed": calls,
            "total_tokens_before":  before,
            "total_tokens_after":   after,
            "tokens_saved":         saved,
            "overall_compression_ratio": round(ratio, 2),
            "average_tokens_before": round(before / max(calls, 1)),
            "average_tokens_after":  round(after  / max(calls, 1)),
        }

    def reset_stats(self):
        self.stats = {
            "total_calls":         0,
            "total_tokens_before": 0,
            "total_tokens_after":  0,
        }