# evaluation/test_conversations.py
#
# The 5 scripted test conversations from the problem statement.
# Each conversation tests a different aspect of context compression.
#
# These are replayed automatically by run_evaluation.py
# No human input needed — fully automated testing.

# ── TEST CONVERSATION A: The Forgotten Allergy ─────────────────
# Tests: Does critical constraint persist across many turns?
# Pass: Agent warns about shellfish at restaurant search
# Fail: Agent recommends seafood without warning

TEST_A_SHELLFISH_ALLERGY = {
    "name": "Test A — The Forgotten Allergy",
    "description": (
        "User states shellfish allergy in turn 1. "
        "After 14 turns of research (flights/hotels/transit), "
        "asks for restaurants near Tsukiji seafood market. "
        "Agent must remember and flag the allergy."
    ),
    "layer": 1,
    "turns": [
        {
            "role": "user",
            "content": (
                "I want to plan a 5-day trip to Tokyo and Kyoto. "
                "My total budget is $3,000. "
                "I am severely allergic to shellfish — "
                "this is a serious medical allergy."
            )
        },
        {
            "role": "user",
            "content": "Search for flights from New York to Tokyo in June"
        },
        {
            "role": "user",
            "content": "What hotels are available in Shinjuku area Tokyo?"
        },
        {
            "role": "user",
            "content": "How do I get from Tokyo to Kyoto? Train options?"
        },
        {
            "role": "user",
            "content": "What is the weather like in Tokyo in June?"
        },
        {
            "role": "user",
            "content": "Find hotels in Kyoto, preferably traditional style"
        },
        {
            "role": "user",
            "content": "What attractions should I visit in Tokyo?"
        },
        {
            "role": "user",
            "content": "How much does the Shinkansen bullet train cost?"
        },
        {
            "role": "user",
            "content": "Find things to do in Kyoto"
        },
        {
            "role": "user",
            "content": "What is the best area to stay in Kyoto?"
        },
        {
            "role": "user",
            "content": "Check the weather in Kyoto for June too"
        },
        {
            "role": "user",
            "content": "How much should I budget for food per day in Japan?"
        },
        {
            "role": "user",
            "content": "What about day trips from Kyoto?"
        },
        {
            "role": "user",
            "content": "Summarize the transportation options between cities"
        },
        {
            "role": "user",
            # THIS IS THE KEY TURN — allergy must be remembered
            "content": (
                "Find me the best dinner spots in the Tsukiji area"
            )
        }
    ],
    "pass_criteria": {
        "turn_index": 14,  # Check the final response (0-indexed)
        "must_contain_any": [
            "shellfish",
            "allergy",
            "seafood",
            "⚠️",
            "warning",
            "allergic",
            "avoid"
        ],
        "must_not_contain": [
            # Agent should not recommend pure seafood without warning
        ],
        "description": (
            "Response must mention shellfish allergy or warn "
            "about seafood when recommending Tsukiji restaurants"
        ),
        "expected_tools": ["places_search"],
        "expected_facts": ["shellfish", "allerg", "budget", "3000"]
    }
}


# ── TEST CONVERSATION B: The Budget Anchor ────────────────────
# Tests: Does numerical state persist and update correctly?
# Pass: Agent knows ~$950 remains and recommends accordingly
# Fail: Agent suggests luxury hotel ignoring budget reality

TEST_B_BUDGET_TRACKING = {
    "name": "Test B — The Budget Anchor",
    "description": (
        "User sets $2,500 budget. "
        "Flights booked at $800, hotels in Rome $400, "
        "hotels in Florence $350. "
        "Agent must know ~$950 remains for Amalfi Coast hotel."
    ),
    "layer": 1,
    "turns": [
        {
            "role": "user",
            "content": (
                "Planning a trip to Italy — Rome, Florence, Amalfi Coast. "
                "7 days total, maximum budget $2,500. "
                "I am a solo traveler."
            )
        },
        {
            "role": "user",
            "content": "Find flights from New York to Rome"
        },
        {
            "role": "user",
            "content": (
                "Book the cheapest flight option. "
                "Let's say we went with $800 for flights. "
                "Update the budget tracker."
            )
        },
        {
            "role": "user",
            "content": "Find hotels in Rome for 2 nights"
        },
        {
            "role": "user",
            "content": (
                "Book a hotel in Rome for $200 per night, 2 nights. "
                "That is $400 total. Update the budget."
            )
        },
        {
            "role": "user",
            "content": "What attractions should I see in Rome?"
        },
        {
            "role": "user",
            "content": "Find hotels in Florence for 2 nights"
        },
        {
            "role": "user",
            "content": (
                "Book a Florence hotel at $175 per night, 2 nights. "
                "That is $350 total. Update the budget."
            )
        },
        {
            "role": "user",
            "content": "What is Florence known for? Attractions?"
        },
        {
            "role": "user",
            "content": "How do I get from Rome to Florence?"
        },
        {
            "role": "user",
            "content": "Tell me about the Amalfi Coast"
        },
        {
            "role": "user",
            "content": "What is the weather like on the Amalfi Coast?"
        },
        {
            "role": "user",
            "content": (
                "Find me a hotel on the Amalfi Coast for 3 nights"
            )
        }
    ],
    "pass_criteria": {
        "turn_index": 12,
        "must_contain_any": [
            "950", "remaining", "budget", "left",
            "950 left", "$950", "within budget", "afford"
        ],
        "must_not_contain_all": [
            # Should not recommend $500/night without flagging budget
        ],
        "description": (
            "Agent must know approximately $950 remains "
            "and recommend hotels within that constraint"
        ),
        "expected_tools": ["places_search"],
        "expected_facts": ["950", "budget"]
    }
}


# ── TEST CONVERSATION C: The Pivot ───────────────────────────
# Tests: Does agent correctly FORGET cancelled plans?
# Pass: Zero Bali references in Switzerland summary
# Fail: Bali bleeds into Switzerland plan

TEST_C_STALE_CONTEXT = {
    "name": "Test C — The Pivot",
    "description": (
        "User plans Bali trip for 6 turns. "
        "Then completely cancels Bali and switches to Switzerland. "
        "Final summary must contain ZERO Bali references."
    ),
    "layer": 2,
    "turns": [
        {
            "role": "user",
            "content": (
                "Plan me a beach vacation in Bali for next month. "
                "Budget $2,000."
            )
        },
        {
            "role": "user",
            "content": "Find flights to Bali"
        },
        {
            "role": "user",
            "content": "Search for beach resorts in Seminyak Bali"
        },
        {
            "role": "user",
            "content": "What surf lessons are available in Bali?"
        },
        {
            "role": "user",
            "content": "Find temple tours in Bali"
        },
        {
            "role": "user",
            "content": "What is the weather in Bali next month?"
        },
        {
            "role": "user",
            # THE PIVOT — Bali must be completely forgotten
            "content": (
                "Actually, scratch Bali entirely. "
                "Forget everything about Bali. "
                "Let us do Switzerland instead — "
                "I want mountains, hiking, not beaches."
            )
        },
        {
            "role": "user",
            "content": "Find flights to Zurich Switzerland"
        },
        {
            "role": "user",
            "content": "Search for mountain hotels in Switzerland"
        },
        {
            "role": "user",
            "content": "What hiking is available in the Swiss Alps?"
        },
        {
            "role": "user",
            "content": "What is the weather in Switzerland?"
        },
        {
            "role": "user",
            "content": "Find restaurants in Zurich"
        },
        {
            "role": "user",
            "content": "What attractions are there in Switzerland?"
        },
        {
            "role": "user",
            "content": "How do I get around Switzerland?"
        },
        {
            "role": "user",
            "content": (
                "Give me a complete summary of my Switzerland trip plan"
            )
        }
    ],
    "pass_criteria": {
        "turn_index": 14,
        "must_not_contain_any": [
            "bali", "beach", "seminyak", "surf",
            "temple tour", "as we discussed about bali",
            "beach resort"
        ],
        "must_contain_any": [
            "switzerland", "zurich", "alps", "mountain", "hiking"
        ],
        "description": (
            "Final summary must contain ZERO Bali references "
            "and focus entirely on Switzerland"
        ),
        "case_sensitive": False,
        "expected_tools": [],  # Summary turn, no tools expected
        "expected_facts": ["switzerland", "zurich", "alps", "mountain", "hiking"]
    }
}


# ── TEST CONVERSATION D: The Logistics Puzzle ─────────────────
# Tests: Can agent connect facts across distant turns?
# Pass: Recommends Thursday train (after Wednesday 2pm meeting)
# Fail: Recommends Wednesday train (ignoring the meeting)

TEST_D_CROSS_REFERENCE = {
    "name": "Test D — The Logistics Puzzle",
    "description": (
        "User mentions Wednesday 2pm meeting in Paris in turn 1. "
        "After 13 turns of research, asks when to take "
        "train to Amsterdam. Agent must cross-reference "
        "turn 1 constraint with current logistics question."
    ),
    "layer": 3,
    "turns": [
        {
            "role": "user",
            "content": (
                "I am planning 6 days: 3 in Paris, 3 in Amsterdam. "
                "I have a work meeting in Paris on Wednesday at 2pm "
                "near the Eiffel Tower. Budget is $2,000."
            )
        },
        {
            "role": "user",
            "content": "Find flights from New York to Paris"
        },
        {
            "role": "user",
            "content": "Search for hotels near the Eiffel Tower in Paris"
        },
        {
            "role": "user",
            "content": "What attractions should I see in Paris?"
        },
        {
            "role": "user",
            "content": "What is the weather like in Paris?"
        },
        {
            "role": "user",
            "content": "Find good restaurants in Paris"
        },
        {
            "role": "user",
            "content": "Tell me about Amsterdam as a destination"
        },
        {
            "role": "user",
            "content": "Find hotels in Amsterdam"
        },
        {
            "role": "user",
            "content": "What is the Anne Frank House experience like?"
        },
        {
            "role": "user",
            "content": "Are canal tours in Amsterdam worth it?"
        },
        {
            "role": "user",
            "content": "What is the weather in Amsterdam?"
        },
        {
            "role": "user",
            "content": "Find restaurants in Amsterdam"
        },
        {
            "role": "user",
            "content": "How do I get from Paris to Amsterdam?"
        },
        {
            "role": "user",
            # KEY TURN — must reference Wednesday meeting from turn 1
            "content": (
                "When should I take the train from Paris to Amsterdam?"
            )
        }
    ],
    "pass_criteria": {
        "turn_index": 13,
        "must_contain_any": [
            "thursday", "wednesday evening", "after your meeting",
            "after the meeting", "wednesday night",
            "meeting", "2pm", "2 pm"
        ],
        "description": (
            "Agent must reason that the Wednesday 2pm meeting "
            "means train should be Thursday morning at earliest. "
            "Must connect turn 1 constraint with turn 13 question."
        ),
        "case_sensitive": False,
        "expected_tools": [],
        "expected_facts": ["meeting", "wednesday", "2pm", "paris"]
    }
}


# ── TEST CONVERSATION E: The Contradiction Detector ──────────
# Tests: Can agent flag conflicts between old constraints and new requests?
# Pass: Agent warns that 15 activities violates max 2/day rule
# Fail: Agent books all 15 activities without flagging conflict

TEST_E_CONTRADICTION = {
    "name": "Test E — The Contradiction Detector",
    "description": (
        "User states max 2 activities per day in turn 1. "
        "After 18 turns of research finding 15+ activities, "
        "user asks to book all of them. "
        "Agent must flag the contradiction."
    ),
    "layer": 3,
    "turns": [
        {
            "role": "user",
            "content": (
                "I want a relaxing trip to Tokyo for 3 days. "
                "No packed schedules please. "
                "Maximum 2 activities per day. "
                "I find busy itineraries stressful."
            )
        },
        {
            "role": "user",
            "content": "Find flights to Tokyo"
        },
        {
            "role": "user",
            "content": "What hotels are in Tokyo?"
        },
        {
            "role": "user",
            "content": "Find attractions in Tokyo"
        },
        {
            "role": "user",
            "content": "Tell me about Senso-ji temple"
        },
        {
            "role": "user",
            "content": "What about Shibuya crossing?"
        },
        {
            "role": "user",
            "content": "Is teamLab Borderless worth visiting?"
        },
        {
            "role": "user",
            "content": "Tell me about Meiji Shrine"
        },
        {
            "role": "user",
            "content": "What about Tokyo Skytree?"
        },
        {
            "role": "user",
            "content": "Find restaurants in Tokyo"
        },
        {
            "role": "user",
            "content": "What about Akihabara district?"
        },
        {
            "role": "user",
            "content": "Tell me about Harajuku"
        },
        {
            "role": "user",
            "content": "What about Shinjuku at night?"
        },
        {
            "role": "user",
            "content": "Is Tsukiji fish market worth visiting?"
        },
        {
            "role": "user",
            "content": "Tell me about Ueno park and museums"
        },
        {
            "role": "user",
            "content": "What about a day trip to Mount Fuji?"
        },
        {
            "role": "user",
            "content": "Any cooking classes in Tokyo?"
        },
        {
            "role": "user",
            "content": "What about sumo wrestling viewing?"
        },
        {
            "role": "user",
            # KEY TURN — must flag contradiction with max 2/day
            "content": (
                "Great! Book all of these activities for my 3-day trip"
            )
        }
    ],
    "pass_criteria": {
        "turn_index": 18,
        "must_contain_any": [
            "2 activities", "two activities", "relaxed",
            "conflict", "maximum", "too many",
            "stressed", "packed", "preference",
            "prioritize", "per day"
        ],
        "description": (
            "Agent must flag that booking all activities "
            "contradicts the max 2 activities per day constraint"
        ),
        "case_sensitive": False,
        "expected_tools": [],
        "expected_facts": ["2", "activities", "per day", "relax"]
    }
}


# All tests in one list for easy iteration
ALL_TESTS = [
    TEST_A_SHELLFISH_ALLERGY,
    TEST_B_BUDGET_TRACKING,
    TEST_C_STALE_CONTEXT,
    TEST_D_CROSS_REFERENCE,
    TEST_E_CONTRADICTION
]