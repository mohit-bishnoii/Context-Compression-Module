# travel_agent/baseline_agent.py
#
# The DUMB agent — no compression whatsoever.
# Stuffs the ENTIRE conversation history into every prompt.

import os
import json
import tiktoken
from groq import Groq
from dotenv import load_dotenv

from travel_agent.tools import (
    web_search,
    places_search,
    weather_fetch,
    budget_tracker,
    reset_budget,
)
from travel_agent.prompts import BASELINE_SYSTEM_PROMPT

load_dotenv()

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search for flights and travel information",
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
            "description": "Search for hotels, restaurants, attractions",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City or area"
                    },
                    "category": {
                        "type": "string",
                        "enum": ["hotels", "restaurants", "attractions"]
                    },
                    "budget_per_night": {
                        "type": "number",
                        "description": "Max price per night (optional, omit if unknown)"
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
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "travel_dates": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "budget_tracker",
            "description": "Track expenses",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["set_budget", "add_expense", "get_status", "reset"]
                    },
                    "amount": {"type": "number"},
                    "category": {"type": "string"},
                    "total_budget": {"type": "number"}
                },
                "required": ["action"]
            }
        }
    }
]


def count_tokens(text: str) -> int:
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(str(text)))
    except Exception:
        return len(str(text)) // 4


def execute_tool(tool_name: str, tool_args: dict) -> str:
    """Execute tool and return RAW uncompressed result as string."""
    try:
        if tool_name == "web_search":
            result = web_search(tool_args["query"])

        elif tool_name == "places_search":
            # Coerce budget_per_night safely
            raw_budget = tool_args.get("budget_per_night")
            budget = None
            if raw_budget is not None:
                try:
                    budget = float(raw_budget)
                except (TypeError, ValueError):
                    budget = None
            result = places_search(
                location=tool_args["location"],
                category=tool_args.get("category", "hotels"),
                budget_per_night=budget,
            )

        elif tool_name == "weather_fetch":
            result = weather_fetch(
                city=tool_args["city"],
                travel_dates=tool_args.get("travel_dates", "upcoming trip"),
            )

        elif tool_name == "budget_tracker":
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
                action=tool_args["action"],
                amount=amount,
                category=tool_args.get("category", "general"),
                total_budget=total_budget,
            )

        else:
            result = {"error": f"Unknown tool: {tool_name}"}

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


BASELINE_TOOL_INSTRUCTION = """

IMPORTANT TOOL USAGE RULES:
- You MUST call tools to answer the user's question.
- User asks about flights → call web_search
- User asks about hotels → call places_search with category="hotels"
- User asks about restaurants → call places_search with category="restaurants"
- User asks about attractions → call places_search with category="attractions"
- User asks about weather → call weather_fetch
- User mentions a budget → call budget_tracker with action="set_budget"
- For places_search: NEVER pass budget_per_night as null or a string. Only include it if you have an explicit numeric value, otherwise omit it entirely.
"""


class BaselineAgent:
    """
    Baseline travel agent with NO context compression.
    Every turn: full conversation history sent to LLM.
    Raw uncompressed tool results added to history.
    """

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model  = "llama-3.3-70b-versatile"
        self.conversation_history   = []
        self.token_counts_per_turn  = []
        self.total_tool_calls       = 0

    def reset(self):
        self.conversation_history   = []
        self.token_counts_per_turn  = []
        self.total_tool_calls       = 0
        reset_budget()
        print("[Baseline] Reset complete")

    def _count_context_tokens(self) -> int:
        total = count_tokens(BASELINE_SYSTEM_PROMPT)
        for msg in self.conversation_history:
            content = msg.get("content", "")
            if content:
                total += count_tokens(str(content))
            if "tool_calls" in msg:
                total += count_tokens(str(msg["tool_calls"]))
        return total

    def chat(self, user_message: str) -> dict:
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
        })

        system_content = BASELINE_SYSTEM_PROMPT + BASELINE_TOOL_INSTRUCTION

        messages = [
            {"role": "system", "content": system_content}
        ] + self.conversation_history

        response_text        = ""
        tool_calls_this_turn = []

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

            max_rounds = 5
            rounds     = 0

            while (
                response.choices[0].finish_reason == "tool_calls"
                and rounds < max_rounds
            ):
                rounds += 1
                tool_calls = response.choices[0].message.tool_calls

                assistant_msg = {
                    "role": "assistant",
                    "content": response.choices[0].message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        }
                        for tc in tool_calls
                    ]
                }
                self.conversation_history.append(assistant_msg)

                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        tool_args = json.loads(tool_call.function.arguments)
                    except Exception:
                        tool_args = {}

                    print(f"  [Baseline] Tool: {tool_name}")
                    raw_result = execute_tool(tool_name, tool_args)
                    self.total_tool_calls += 1

                    tool_calls_this_turn.append({
                        "tool": tool_name,
                        "raw_result_tokens": count_tokens(raw_result),
                    })

                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": raw_result,
                    })

                messages = [
                    {"role": "system", "content": system_content}
                ] + self.conversation_history

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

        except Exception as e:
            error_msg = str(e)
            if "413" in error_msg or "too large" in error_msg.lower() or "rate_limit" in error_msg.lower():
                response_text = (
                    "CONTEXT OVERFLOW ERROR: "
                    "Too much history for the model to process. "
                    "This is the failure that CCM prevents."
                )
            else:
                response_text = f"Error: {error_msg}"
            print(f"[Baseline] Error: {e}")

        self.conversation_history.append({
            "role": "assistant",
            "content": response_text,
        })

        tokens_after = self._count_context_tokens()
        self.token_counts_per_turn.append(tokens_after)

        return {
            "response": response_text,
            "tokens_in_context": tokens_after,
            "tool_calls": tool_calls_this_turn,
            "turn_number": len(self.token_counts_per_turn),
            "agent_type": "baseline",
        }

    def get_metrics(self) -> dict:
        return {
            "agent_type": "baseline",
            "total_turns": len(self.token_counts_per_turn),
            "token_counts_per_turn": self.token_counts_per_turn,
            "max_tokens_used": (
                max(self.token_counts_per_turn) if self.token_counts_per_turn else 0
            ),
            "avg_tokens_per_turn": (
                sum(self.token_counts_per_turn) // len(self.token_counts_per_turn)
                if self.token_counts_per_turn else 0
            ),
            "total_tool_calls": self.total_tool_calls,
        }