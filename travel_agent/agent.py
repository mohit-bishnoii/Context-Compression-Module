# travel_agent/agent.py
#
# The CCM-powered travel agent.

import os
import json
import tiktoken
from groq import Groq
from dotenv import load_dotenv

from ccm.ccm_core import ContextCompressionModule
from travel_agent.tools import (
    web_search,
    places_search,
    weather_fetch,
    budget_tracker,
    reset_budget,
)
from travel_agent.prompts import TRAVEL_AGENT_SYSTEM_PROMPT

load_dotenv()

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search for flights, trains, and general travel information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "places_search",
            "description": "Search for hotels, restaurants, or attractions in a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City or area to search in"
                    },
                    "category": {
                        "type": "string",
                        "description": "Type of place to search for",
                        "enum": ["hotels", "restaurants", "attractions"]
                    },
                    "budget_per_night": {
                        "type": "number",
                        "description": "Max price per night in USD. Only include when user stated a specific number."
                    }
                },
                "required": ["location", "category"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "weather_fetch",
            "description": "Get weather forecast and packing advice for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name"
                    },
                    "travel_dates": {
                        "type": "string",
                        "description": "Travel month or date range"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "budget_tracker",
            "description": "Track travel expenses and remaining budget",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["set_budget", "add_expense", "get_status", "reset"]
                    },
                    "amount": {
                        "type": "number",
                        "description": "Expense amount in USD"
                    },
                    "category": {
                        "type": "string",
                        "description": "Expense category e.g. flights, hotels"
                    },
                    "total_budget": {
                        "type": "number",
                        "description": "Total trip budget for set_budget action"
                    }
                },
                "required": ["action"]
            }
        }
    }
]


def count_tokens(text: str) -> int:
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4


def _is_tool_call_error(error_str: str) -> bool:
    """Detect the malformed function call error from Groq."""
    return (
        "tool_use_failed" in error_str
        or "failed_generation" in error_str
        or "Failed to call a function" in error_str
    )


def _format_result_for_llm(tool_name: str, raw_result: dict) -> str:
    try:
        if tool_name == "places_search":
            results = raw_result.get("results", [])
            if not results:
                return f"places_search: {raw_result.get('message', 'No results found.')}"
            lines = []
            for r in results[:5]:
                name    = r.get("name") or "Unknown"
                addr    = r.get("address") or r.get("formatted") or ""
                price   = r.get("price_per_night") or r.get("price_range") or ""
                rating  = r.get("rating") or ""
                warning = r.get("allergy_warning") or ""
                line    = f"- {name}"
                if addr:   line += f", {addr}"
                if price:  line += f" (${price}/night)"
                if rating: line += f" star:{rating}"
                if warning: line += f" | WARNING: {warning}"
                lines.append(line)
            return f"Places ({raw_result.get('data_source','')}):\n" + "\n".join(lines)

        elif tool_name == "web_search":
            results = raw_result.get("results", [])
            if not results:
                return f"web_search: {raw_result.get('message', 'No results.')}"
            lines = []
            for r in results[:5]:
                title   = r.get("title") or r.get("airline") or ""
                snippet = r.get("snippet") or ""
                price   = r.get("price") or ""
                line = f"- {title}"
                if price:   line += f": {price}"
                if snippet: line += f" - {snippet}"
                lines.append(line)
            return f"Search [{raw_result.get('search_type','')}]:\n" + "\n".join(lines)

        elif tool_name == "weather_fetch":
            city  = raw_result.get("city", "")
            cond  = raw_result.get("current_conditions", {})
            lines = [f"Weather {city}: {cond.get('temperature_f','?')}F, {cond.get('description','')}"]
            note = raw_result.get("seasonal_note", "")
            if note: lines.append(f"Note: {note}")
            packing = raw_result.get("packing_recommendations", [])
            if packing: lines.append(f"Pack: {', '.join(packing[:6])}")
            return "\n".join(lines)

        elif tool_name == "budget_tracker":
            total     = raw_result.get("total_budget", "")
            spent     = raw_result.get("total_spent", raw_result.get("amount_spent", ""))
            remaining = raw_result.get("remaining", "")
            warning   = raw_result.get("warning", "")
            lines = [f"Budget status: {raw_result.get('status','')}"]
            if total != "":
                lines.append(f"Total: ${total} | Spent: ${spent} | Remaining: ${remaining}")
            if warning: lines.append(warning)
            return "\n".join(lines)

        else:
            raw_str = json.dumps(raw_result, indent=2)
            return raw_str[:500] + ("..." if len(raw_str) > 500 else "")

    except Exception:
        return str(raw_result)[:400]


def execute_tool(tool_name: str, tool_args: dict) -> tuple:
    """Execute a tool with safe type coercion."""
    try:
        if tool_name == "web_search":
            query = str(tool_args.get("query", ""))
            return web_search(query), query

        elif tool_name == "places_search":
            location = str(tool_args.get("location", ""))
            category = str(tool_args.get("category", "hotels"))
            raw_budget = tool_args.get("budget_per_night")
            budget = None
            if raw_budget is not None:
                try:
                    budget = float(raw_budget)
                except (TypeError, ValueError):
                    budget = None
            result = places_search(location, category, budget)
            return result, f"{category} in {location}"

        elif tool_name == "weather_fetch":
            city  = str(tool_args.get("city", ""))
            dates = str(tool_args.get("travel_dates", "upcoming trip"))
            return weather_fetch(city, dates), f"weather in {city}"

        elif tool_name == "budget_tracker":
            def _to_float(v):
                try: return float(v) if v else 0
                except (TypeError, ValueError): return 0
            result = budget_tracker(
                action=tool_args.get("action"),
                amount=_to_float(tool_args.get("amount")),
                category=str(tool_args.get("category", "general")),
                total_budget=_to_float(tool_args.get("total_budget")),
            )
            return result, f"budget {tool_args.get('action','')}"

        else:
            return {"error": f"Unknown tool: {tool_name}"}, tool_name

    except Exception as e:
        return {"error": str(e)}, tool_name


def _looks_like_food_query(msg: str) -> bool:
    kw = ["restaurant", "restaurants", "dinner", "lunch", "breakfast",
          "food", "eat", "dining", "spots", "cafe"]
    return any(w in msg.lower() for w in kw)


def _response_mentions_allergy(text: str) -> bool:
    markers = ["shellfish", "allerg", "seafood", "warning", "avoid",
               "safe for", "not suitable", "cannot eat"]
    return any(m in text.lower() for m in markers)


def _enforce_critical_constraints(response_text, user_message, tool_calls_this_turn, memory_state):
    critical_facts = (
        memory_state.get("working_memory_raw", {})
        .get("facts", {}).get("critical", [])
    )
    critical_values = [str(f.get("value", "")).lower() for f in critical_facts]
    has_shellfish = any("shellfish" in v or "allerg" in v for v in critical_values)

    if not has_shellfish or not _looks_like_food_query(user_message):
        return response_text
    if _response_mentions_allergy(response_text):
        return response_text

    safe, caution, unsafe = [], [], []
    for call in tool_calls_this_turn:
        if call.get("tool") != "places_search":
            continue
        for item in call.get("raw_result", {}).get("results", []):
            if not isinstance(item, dict): continue
            name = item.get("name")
            if not name: continue
            w = str(item.get("allergy_warning", "")).lower()
            c = str(item.get("cuisine", "")).lower()
            if any(m in w for m in ["safe for shellfish", "shellfish-free", "no shellfish"]):
                safe.append(name)
            elif "shellfish" in w or ("seafood" in c and "safe" not in w):
                unsafe.append(name)
            elif any(m in w for m in ["check with staff", "ask vendor"]):
                caution.append(name)

    parts = ["WARNING: Due to your severe shellfish allergy, be careful at seafood-heavy spots."]
    if unsafe:  parts.append(f"Avoid: {', '.join(list(dict.fromkeys(unsafe))[:2])}.")
    if safe:    parts.append(f"Safer options: {', '.join(list(dict.fromkeys(safe))[:2])}.")
    elif caution: parts.append(f"Ask staff about allergens at: {', '.join(list(dict.fromkeys(caution))[:2])}.")

    prefix = " ".join(parts)
    return f"{prefix}\n\n{response_text}" if response_text.strip() else prefix


def _build_clean_context(compressed_context: str, user_message: str) -> str:
    """
    Build a clean user message that won't confuse the LLM's tool-call generation.
    The key fix: avoid putting angle-bracket-heavy content right before tool calls.
    We wrap the context in a clearly labeled block.
    """
    return (
        "MEMORY CONTEXT (use this to inform your response):\n"
        + compressed_context.strip()
        + "\n\nUSER REQUEST: "
        + user_message
    )


class CCMAgent:
    """Travel agent powered by the Context Compression Module."""

    def __init__(self, use_reranking: bool = True):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model  = "llama-3.3-70b-versatile"
        self.ccm    = ContextCompressionModule(use_reranking=use_reranking)
        self.turn_count     = 0
        self.token_counts   = []
        self.tool_calls_log = []

    def reset(self):
        self.ccm.reset()
        reset_budget()
        self.turn_count     = 0
        self.token_counts   = []
        self.tool_calls_log = []

    def _call_llm(self, messages: list, use_tools: bool = True):
        kwargs = dict(
            model=self.model,
            messages=messages,
            max_tokens=1024,
            temperature=0.0,
        )
        if use_tools:
            kwargs["tools"] = TOOL_DEFINITIONS
            kwargs["tool_choice"] = "auto"
            # CHANGE THIS TO TRUE
            kwargs["parallel_tool_calls"] = True 
        return self.client.chat.completions.create(**kwargs)

    def chat(self, user_message: str) -> dict:
        self.turn_count += 1
        tool_calls_this_turn = []

        # Get compressed context from CCM
        compressed_context = self.ccm.process_user_message(user_message)

        # Build clean user content — avoid format that confuses tool-call generation
        user_content = _build_clean_context(compressed_context, user_message)

        messages = [
            {"role": "system", "content": TRAVEL_AGENT_SYSTEM_PROMPT},
            {"role": "user",   "content": user_content}
        ]

        context_tokens = count_tokens(TRAVEL_AGENT_SYSTEM_PROMPT + compressed_context)
        response_text  = ""

        try:
            response = self._call_llm(messages, use_tools=True)

            max_tool_rounds = 2
            tool_round      = 0

            while (
                response.choices[0].finish_reason == "tool_calls"
                and tool_round < max_tool_rounds
            ):
                tool_round += 1
                tool_calls = response.choices[0].message.tool_calls

                messages.append({
                    "role": "assistant",
                    "content": response.choices[0].message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in tool_calls
                    ]
                })

                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        tool_args = json.loads(tool_call.function.arguments)
                    except Exception:
                        tool_args = {}

                    print(f"  [CCMAgent] Tool: {tool_name}({tool_args})")
                    raw_result, query_used = execute_tool(tool_name, tool_args)

                    compressed_result = self.ccm.process_tool_result(
                        tool_name=tool_name,
                        raw_result=raw_result,
                        query_used=query_used,
                    )

                    tool_calls_this_turn.append({
                        "tool": tool_name,
                        "args": tool_args,
                        "result_preview": compressed_result[:100],
                        "compressed_result": compressed_result,
                        "raw_result": raw_result,
                    })
                    self.tool_calls_log.append({
                        "turn": self.turn_count,
                        "tool": tool_name,
                        "query": query_used,
                    })

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": _format_result_for_llm(tool_name, raw_result),
                    })

                response = self._call_llm(messages, use_tools=True)

            response_text = response.choices[0].message.content or ""

        except Exception as e:
            error_str = str(e)
            print(f"[CCMAgent] Error: {e}")

            # ── Retry without tools if malformed tool-call error ──
            if _is_tool_call_error(error_str):
                print("[CCMAgent] Retrying without tools (tool_use_failed)...")
                try:
                    retry_messages = [
                        {"role": "system", "content": TRAVEL_AGENT_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                user_content
                                + "\n\nNote: Tool calls are unavailable right now. "
                                "Answer based on your knowledge and the memory context above."
                            )
                        }
                    ]
                    retry_response = self._call_llm(retry_messages, use_tools=False)
                    response_text = retry_response.choices[0].message.content or ""
                    print("[CCMAgent] Retry succeeded.")
                except Exception as retry_err:
                    response_text = f"I encountered an error processing your request: {retry_err}"
            else:
                response_text = f"I encountered an error: {error_str}"

        # Apply allergy safety guardrail
        response_text = _enforce_critical_constraints(
            response_text=response_text,
            user_message=user_message,
            tool_calls_this_turn=tool_calls_this_turn,
            memory_state=self.ccm.get_memory_state(),
        )

        # Update CCM memory
        self.ccm.process_agent_response(
            user_message=user_message,
            agent_response=response_text,
            tool_calls_made=tool_calls_this_turn,
        )

        self.token_counts.append(context_tokens)

        return {
            "response": response_text,
            "tokens_in_context": context_tokens,
            "tool_calls": tool_calls_this_turn,
            "turn_number": self.turn_count,
            "agent_type": "ccm",
            "memory_state": self.ccm.get_memory_state(),
        }

    def get_metrics(self) -> dict:
        memory_state = self.ccm.get_memory_state()
        return {
            "agent_type": "ccm",
            "total_turns": self.turn_count,
            "token_counts_per_turn": self.token_counts,
            "max_tokens_used": max(self.token_counts) if self.token_counts else 0,
            "avg_tokens_per_turn": (
                sum(self.token_counts) // len(self.token_counts)
                if self.token_counts else 0
            ),
            "total_tool_calls": len(self.tool_calls_log),
            "compression_stats": memory_state["compression_stats"],
            "token_metrics": memory_state["token_metrics"],
        }