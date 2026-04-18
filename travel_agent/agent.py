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

# Tool definitions — budget_per_night is optional and uses oneOf to allow null
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search for flights, trains, and general travel information"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
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
            "description": (
                "Search for hotels, restaurants, or attractions"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City or area to search in"
                    },
                    "category": {
                        "type": "string",
                        "description": "hotels, restaurants, or attractions",
                        "enum": ["hotels", "restaurants", "attractions"]
                    },
                    "budget_per_night": {
                        "type": "number",
                        "description": "Max price per night for hotels (optional, omit if not known)"
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
            "description": (
                "Get weather and packing recommendations for a city"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name"
                    },
                    "travel_dates": {
                        "type": "string",
                        "description": "Travel dates or month"
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
            "description": "Track travel expenses and check budget",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "set_budget",
                            "add_expense",
                            "get_status",
                            "reset"
                        ]
                    },
                    "amount": {
                        "type": "number",
                        "description": "Expense amount"
                    },
                    "category": {
                        "type": "string",
                        "description": "Expense category"
                    },
                    "total_budget": {
                        "type": "number",
                        "description": "Total budget for set_budget action"
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


def _format_result_for_llm(tool_name: str, raw_result: dict) -> str:
    try:
        if tool_name == "places_search":
            results = raw_result.get("results", [])
            if not results:
                msg = raw_result.get("message", "No results found.")
                return f"places_search: {msg}"
            lines = []
            for r in results[:5]:
                name    = r.get("name") or "Unknown"
                addr    = r.get("address") or r.get("formatted") or ""
                price   = r.get("price_per_night") or r.get("price_range") or ""
                rating  = r.get("rating") or ""
                warning = r.get("allergy_warning") or ""
                line    = f"- {name}"
                if addr:
                    line += f", {addr}"
                if price:
                    line += f" (${price}/night)"
                if rating:
                    line += f" ★{rating}"
                if warning:
                    line += f" | ⚠️ {warning}"
                lines.append(line)
            source = raw_result.get("data_source", "")
            return f"Places results ({source}):\n" + "\n".join(lines)

        elif tool_name == "web_search":
            results = raw_result.get("results", [])
            if not results:
                msg = raw_result.get("message", "No results found.")
                return f"web_search: {msg}"
            lines = []
            for r in results[:5]:
                title   = r.get("title") or r.get("airline") or ""
                snippet = r.get("snippet") or ""
                price   = r.get("price") or ""
                line = f"- {title}"
                if price:
                    line += f": {price}"
                if snippet:
                    line += f" — {snippet}"
                lines.append(line)
            stype  = raw_result.get("search_type", "")
            return f"Search results [{stype}]:\n" + "\n".join(lines)

        elif tool_name == "weather_fetch":
            city  = raw_result.get("city", "")
            cond  = raw_result.get("current_conditions", {})
            temp  = cond.get("temperature_f", "?")
            desc  = cond.get("description", "")
            note  = raw_result.get("seasonal_note", "")
            packing = raw_result.get("packing_recommendations", [])
            lines = [f"Weather in {city}: {temp}°F, {desc}"]
            if note:
                lines.append(f"Seasonal note: {note}")
            if packing:
                lines.append(f"Pack: {', '.join(packing[:6])}")
            return "\n".join(lines)

        elif tool_name == "budget_tracker":
            status    = raw_result.get("status", "")
            total     = raw_result.get("total_budget", "")
            spent     = raw_result.get("total_spent", raw_result.get("amount_spent", ""))
            remaining = raw_result.get("remaining", "")
            warning   = raw_result.get("warning", "")
            lines = [f"Budget: {status}"]
            if total != "":
                lines.append(f"Total: ${total}  |  Spent: ${spent}  |  Remaining: ${remaining}")
            if warning:
                lines.append(warning)
            return "\n".join(lines)

        else:
            raw_str = json.dumps(raw_result, indent=2)
            return raw_str[:600] + ("…" if len(raw_str) > 600 else "")

    except Exception:
        return str(raw_result)[:400]


def execute_tool(tool_name: str, tool_args: dict) -> tuple:
    """Execute a tool, coercing types to avoid validation errors."""
    try:
        if tool_name == "web_search":
            query = tool_args.get("query", "")
            result = web_search(query)
            return result, query

        elif tool_name == "places_search":
            location = tool_args.get("location", "")
            category = tool_args.get("category", "hotels")
            # Safely coerce budget_per_night — LLM sometimes sends null/string
            raw_budget = tool_args.get("budget_per_night")
            budget = None
            if raw_budget is not None:
                try:
                    budget = float(raw_budget)
                except (TypeError, ValueError):
                    budget = None
            result = places_search(location, category, budget)
            query = f"{category} in {location}"
            return result, query

        elif tool_name == "weather_fetch":
            city = tool_args.get("city", "")
            dates = tool_args.get("travel_dates", "upcoming trip")
            result = weather_fetch(city, dates)
            query = f"weather in {city}"
            return result, query

        elif tool_name == "budget_tracker":
            # Coerce numeric fields
            raw_amount = tool_args.get("amount", 0)
            raw_total  = tool_args.get("total_budget", 0)
            try:
                amount = float(raw_amount) if raw_amount else 0
            except (TypeError, ValueError):
                amount = 0
            try:
                total_budget = float(raw_total) if raw_total else 0
            except (TypeError, ValueError):
                total_budget = 0

            result = budget_tracker(
                action=tool_args.get("action"),
                amount=amount,
                category=tool_args.get("category", "general"),
                total_budget=total_budget,
            )
            query = f"budget {tool_args.get('action', '')}"
            return result, query

        else:
            return {"error": f"Unknown tool: {tool_name}"}, tool_name

    except Exception as e:
        return {"error": str(e)}, tool_name


def _looks_like_food_query(user_message: str) -> bool:
    msg = user_message.lower()
    keywords = [
        "restaurant", "restaurants", "dinner", "lunch", "breakfast",
        "food", "eat", "dining", "spots", "cafe",
    ]
    return any(word in msg for word in keywords)


def _response_mentions_allergy(response_text: str) -> bool:
    text = response_text.lower()
    markers = [
        "shellfish", "allerg", "seafood", "warning", "avoid",
        "safe for", "not suitable", "cannot eat",
    ]
    return any(marker in text for marker in markers)


def _classify_restaurant_safety(item: dict) -> str:
    warning = str(item.get("allergy_warning", "")).lower()
    cuisine = str(item.get("cuisine", "")).lower()

    safe_markers = [
        "safe for shellfish allergy", "shellfish-free",
        "no shellfish", "accommodate shellfish allergy",
        "allergy accommodations available",
    ]
    caution_markers = [
        "check with staff", "ask vendor", "ask staff",
        "advance notice",
    ]

    if any(marker in warning for marker in safe_markers):
        return "safe"
    if (
        "seafood" in cuisine and "safe" not in warning
    ) or "not suitable for shellfish allergy" in warning or "primary ingredient" in warning:
        return "unsafe"
    if any(marker in warning for marker in caution_markers):
        return "caution"
    if "shellfish" in warning:
        return "unsafe"
    return "unknown"


def _enforce_critical_constraints(
    response_text: str,
    user_message: str,
    tool_calls_this_turn: list,
    memory_state: dict,
) -> str:
    critical_facts = (
        memory_state.get("working_memory_raw", {})
        .get("facts", {})
        .get("critical", [])
    )
    critical_values = [
        str(fact.get("value", "")).lower()
        for fact in critical_facts
    ]

    has_shellfish_constraint = any(
        "shellfish" in value or "allerg" in value
        for value in critical_values
    )
    if not has_shellfish_constraint or not _looks_like_food_query(user_message):
        return response_text

    if _response_mentions_allergy(response_text):
        return response_text

    safe_names    = []
    caution_names = []
    unsafe_names  = []

    for call in tool_calls_this_turn:
        if call.get("tool") != "places_search":
            continue
        raw_result = call.get("raw_result", {})
        if not isinstance(raw_result, dict):
            continue
        for item in raw_result.get("results", []):
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not name:
                continue
            safety = _classify_restaurant_safety(item)
            if safety == "safe":
                safe_names.append(name)
            elif safety == "caution":
                caution_names.append(name)
            elif safety == "unsafe":
                unsafe_names.append(name)

    safe_names    = list(dict.fromkeys(safe_names))
    caution_names = list(dict.fromkeys(caution_names))
    unsafe_names  = list(dict.fromkeys(unsafe_names))

    warning_lines = [
        "⚠️ Because of your severe shellfish allergy, I would avoid seafood-heavy spots and double-check ingredients with staff."
    ]
    if unsafe_names:
        warning_lines.append("Avoid: " + ", ".join(unsafe_names[:2]) + ".")
    if safe_names:
        warning_lines.append("Safer picks: " + ", ".join(safe_names[:2]) + ".")
    elif caution_names:
        warning_lines.append(
            "More allergy-aware options: " + ", ".join(caution_names[:2]) + "."
        )

    warning_prefix = " ".join(warning_lines)
    if response_text.strip():
        return f"{warning_prefix}\n\n{response_text}"
    return warning_prefix


class CCMAgent:
    """Travel agent powered by the Context Compression Module."""

    def __init__(self, use_reranking: bool = True):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model  = "llama-3.3-70b-versatile"
        self.ccm    = ContextCompressionModule(use_reranking=use_reranking)
        self.turn_count  = 0
        self.token_counts = []
        self.tool_calls_log = []

    def reset(self):
        self.ccm.reset()
        reset_budget()
        self.turn_count   = 0
        self.token_counts = []
        self.tool_calls_log = []

    def chat(self, user_message: str) -> dict:
        self.turn_count += 1
        tool_calls_this_turn = []

        # Get compressed context from CCM
        compressed_context = self.ccm.process_user_message(user_message)

        # Build the system prompt — explicitly tell the agent to USE tools
        system_prompt = TRAVEL_AGENT_SYSTEM_PROMPT + """

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPORTANT TOOL USAGE RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You MUST call tools to answer the user's question. Do NOT just ask
"what would you like to do next?" without first using a tool.

- User asks about flights → call web_search
- User asks about hotels → call places_search with category="hotels"
- User asks about restaurants → call places_search with category="restaurants"
- User asks about attractions → call places_search with category="attractions"
- User asks about weather → call weather_fetch
- User mentions a budget → call budget_tracker with action="set_budget"

IMPORTANT: For places_search, NEVER pass budget_per_night as null or a string.
Only include budget_per_night if you have an explicit numeric value.
If no budget is specified for hotels, omit budget_per_night entirely.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"{compressed_context}\n\n"
                    f"[CURRENT MESSAGE]\n{user_message}"
                )
            }
        ]

        context_tokens = count_tokens(system_prompt + compressed_context)
        response_text  = ""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                max_tokens=1024,
                temperature=0.0,
                parallel_tool_calls=False,
            )

            max_tool_rounds = 5
            tool_round = 0

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

                    print(f"  [CCMAgent] Tool call: {tool_name}({tool_args})")

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

                    llm_tool_content = _format_result_for_llm(tool_name, raw_result)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": llm_tool_content,
                    })

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                    tool_choice="auto",
                    max_tokens=1024,
                    temperature=0.0,
                    parallel_tool_calls=False,
                )

            response_text = response.choices[0].message.content or ""
            response_text = _enforce_critical_constraints(
                response_text=response_text,
                user_message=user_message,
                tool_calls_this_turn=tool_calls_this_turn,
                memory_state=self.ccm.get_memory_state(),
            )

        except Exception as e:
            response_text = f"I encountered an error: {str(e)}"
            print(f"[CCMAgent] Error: {e}")

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