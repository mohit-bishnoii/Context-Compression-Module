# travel_agent/baseline_agent.py
#
# The DUMB agent — no compression whatsoever.
# Stuffs the ENTIRE conversation history into every prompt.
# Includes raw uncompressed tool results.
#
# PURPOSE:
#   This is the BEFORE state.
#   We run this first to PROVE the problem exists.
#   Then we run CCMAgent to show the fix.
#
# WHAT WILL GO WRONG:
#   Turn 1-5:   Works fine
#   Turn 6-12:  Starts getting slow (more tokens each turn)
#   Turn 13-16: May lose early constraints (allergy forgotten)
#   Turn 17+:   Likely to hit 8K context limit and fail entirely

import os
import json
import time
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
                        "description": (
                            "Max price per night — for hotels ONLY. "
                            "Do NOT include this parameter when searching "
                            "restaurants or attractions."
                        )
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
                        "enum": [
                            "set_budget", "add_expense",
                            "get_status", "reset"
                        ]
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
            result = places_search(
                location=tool_args["location"],
                category=tool_args.get("category", "hotels"),
                budget_per_night=tool_args.get("budget_per_night")
            )
        elif tool_name == "weather_fetch":
            result = weather_fetch(
                city=tool_args["city"],
                travel_dates=tool_args.get(
                    "travel_dates", "upcoming trip"
                )
            )
        elif tool_name == "budget_tracker":
            result = budget_tracker(
                action=tool_args["action"],
                amount=tool_args.get("amount", 0),
                category=tool_args.get("category", "general"),
                total_budget=tool_args.get("total_budget", 0)
            )
        else:
            result = {"error": f"Unknown tool: {tool_name}"}

        # Return FULL uncompressed JSON — this is intentional
        # This is what causes context overflow in baseline
        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


class BaselineAgent:
    """
    Baseline travel agent with NO context compression.

    Every turn:
    - Full conversation history sent to LLM
    - Raw uncompressed tool results added to history
    - No memory extraction
    - No stale context detection
    - No RAG retrieval

    This agent WILL fail on long conversations.
    That failure is the proof that CCM is needed.
    """

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.1-8b-instant"
        self.conversation_history = []
        self.token_counts_per_turn = []
        self.total_tool_calls = 0
        self.latency_per_turn: list[float] = []
        self._llm_call_count = 0

    def reset(self):
        """Reset for new conversation."""
        self.conversation_history = []
        self.token_counts_per_turn = []
        self.total_tool_calls = 0
        self.latency_per_turn = []
        self._llm_call_count = 0
        reset_budget()
        print("[Baseline] Reset complete")

    def _count_context_tokens(self) -> int:
        """Count total tokens in current full history."""
        total = count_tokens(BASELINE_SYSTEM_PROMPT)
        for msg in self.conversation_history:
            content = msg.get("content", "")
            if content:
                total += count_tokens(str(content))
            # Count tool calls if present
            if "tool_calls" in msg:
                total += count_tokens(str(msg["tool_calls"]))
        return total

    def chat(self, user_message: str) -> dict:
        """
        Send message, get response.
        NO compression — full history every time.

        Returns same format as CCMAgent for easy comparison.
        """
        # Add to full history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        _turn_start = time.perf_counter()

        # Build messages — FULL HISTORY every turn
        # This grows unboundedly — the core problem
        messages = [
            {"role": "system", "content": BASELINE_SYSTEM_PROMPT}
        ] + self.conversation_history

        tokens_before = self._count_context_tokens()
        response_text = ""
        tool_calls_this_turn = []

        try:
            _t0 = time.perf_counter()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                max_tokens=1024,
                temperature=0.0,
                parallel_tool_calls=False
            )
            self._llm_call_count += 1

            # Handle tool calls
            max_rounds = 5
            rounds = 0

            while (
                response.choices[0].finish_reason == "tool_calls"
                and rounds < max_rounds
            ):
                rounds += 1
                tool_calls = response.choices[0].message.tool_calls

                # Add assistant tool call message to history
                assistant_msg = {
                    "role": "assistant",
                    "content": (
                        response.choices[0].message.content or ""
                    ),
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
                }
                self.conversation_history.append(assistant_msg)

                # Execute tools and add RAW results to history
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(
                        tool_call.function.arguments
                    )
                    # Strip null values — Groq rejects tool calls where an
                    # optional numeric field (e.g. budget_per_night) is null.
                    tool_args = {
                        k: v for k, v in tool_args.items()
                        if v is not None
                    }

                    print(
                        f"  [Baseline] Tool: {tool_name}"
                    )

                    # RAW result — no compression
                    raw_result = execute_tool(tool_name, tool_args)
                    self.total_tool_calls += 1

                    raw_tokens = count_tokens(raw_result)
                    tool_calls_this_turn.append({
                        "tool": tool_name,
                        "raw_result_tokens": raw_tokens
                    })

                    # Add FULL raw result to history
                    # This is what causes overflow
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": raw_result
                    })

                # Rebuild full messages and call again
                messages = [
                    {
                        "role": "system",
                        "content": BASELINE_SYSTEM_PROMPT
                    }
                ] + self.conversation_history

                _t0 = time.perf_counter()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                    tool_choice="auto",
                    max_tokens=1024,
                    temperature=0.0,
                    parallel_tool_calls=False
                )
                self._llm_call_count += 1

            response_text = (
                response.choices[0].message.content or ""
            )

        except Exception as e:
            error_msg = str(e)
            response_text = f"Error: {error_msg}"

            # Context overflow shows up as this error
            if "413" in error_msg or "too large" in error_msg.lower():
                response_text = (
                    "CONTEXT OVERFLOW ERROR: "
                    "Too much history for the model to process. "
                    "This is the failure that CCM prevents."
                )
            print(f"[Baseline] Error: {e}")

        # Add response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response_text
        })

        # Count tokens AFTER (includes response)
        tokens_after = self._count_context_tokens()
        self.token_counts_per_turn.append(tokens_after)
        self.latency_per_turn.append(time.perf_counter() - _turn_start)

        return {
            "response": response_text,
            "tokens_in_context": tokens_after,
            "tool_calls": tool_calls_this_turn,
            "turn_number": len(self.token_counts_per_turn),
            "agent_type": "baseline"
        }

    def get_metrics(self) -> dict:
        avg_lat = (
            sum(self.latency_per_turn) / len(self.latency_per_turn)
            if self.latency_per_turn else 0.0
        )
        return {
            "agent_type": "baseline",
            "total_turns": len(self.token_counts_per_turn),
            "token_counts_per_turn": self.token_counts_per_turn,
            "max_tokens_used": (
                max(self.token_counts_per_turn)
                if self.token_counts_per_turn else 0
            ),
            "avg_tokens_per_turn": (
                sum(self.token_counts_per_turn) //
                len(self.token_counts_per_turn)
                if self.token_counts_per_turn else 0
            ),
            "total_tool_calls": self.total_tool_calls,
            # Latency
            "latency_per_turn": self.latency_per_turn,
            "avg_latency_per_turn_s": round(avg_lat, 3),
            "total_latency_s": round(sum(self.latency_per_turn), 3),
            "total_llm_calls": self._llm_call_count,
        }
