# ccm/topic_tracker.py
#
# Topic-based memory system for episodic summarization.
#
# Replaces fixed 4-turn summarization with topic-based:
#   - Extract topic from user message
#   - Detect conclusion signals
#   - Summarize only when topic is concluded
#   - Support dynamic topic switching

from typing import Optional

TOPIC_KEYWORDS = {
    "flights": ["flight", "flights", "fly", "flying", "airline", "airport", "depart", "arrival", "direct", "connecting"],
    "hotels": ["hotel", "hotels", "stay", "staying", "accommodation", "lodging", "room", "resort", "inn", "booking"],
    "restaurants": ["restaurant", "restaurants", "food", "eat", "dinner", "lunch", "breakfast", "dining", "cafe", "bar", "spots", "meal"],
    "weather": ["weather", "temperature", "climate", "rainy", "sunny", "hot", "cold", "humid", "forecast", "packing"],
    "budget": ["budget", "cost", "price", "expensive", "cheap", "afford", "spend", "money", "fare", "fee"],
    "visa": ["visa", "passport", "entry", "immigration", "customs", "documents", "travel requirement"],
    "activities": ["activities", "activity", "tour", "tourist", "attraction", "sightseeing", "visit", "see", "do"],
    "transportation": ["transport", "transportation", "train", "taxi", "uber", "subway", "bus", "rail", "transit", "car rental"],
}

CONCLUSION_KEYWORDS = [
    "okay", "okay let's", "okay let us", "let's do it", "let's book", "book it",
    "perfect", "that's perfect", "sounds great",
    "i'll take it", "i'll take", "i will take it", "i will take", "go with that",
    "that works", "confirmed", "done", "done deal", "yes please", "yes please book",
    "go ahead", "proceed", "good enough", "i like it", "i like that",
    "that's the one", "we'll take it", "we will take it", "we'll go with", "we will go with",
    "decided", "decided on", "lets go with it", "lets book it", "lets proceed",
    "sounds good",
]

IMPLICIT_CONCLUSION_KEYWORDS = [
    "actually", "instead", "let's do a", "let's try", "new topic",
    "forget that", "never mind", "different destination", "change topic", "switch to",
]

def extract_topic(message: str) -> str:
    """
    Extract topic from user message using keyword matching.
    Uses word boundary matching to avoid substring matches.

    Returns: topic string (e.g., "flights", "hotels", "restaurants", "weather", etc.)
             or "general" if no match
    """
    import re
    message_lower = message.lower()

    for topic, keywords in TOPIC_KEYWORDS.items():
        for keyword in keywords:
            # Use word boundary matching to avoid substring issues
            # e.g., "weather" shouldn't match "the weath er"
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, message_lower):
                return topic

    return "general"


def detect_explicit_conclusion(message: str) -> bool:
    """
    Detect explicit conclusion signal in user message.
    User is indicating topic is resolved/ready to move on.
    
    Returns: True if conclusion signal detected
    """
    message_lower = message.lower().strip()
    
    for keyword in CONCLUSION_KEYWORDS:
        if keyword in message_lower:
            return True
    
    return False


def detect_implicit_conclusion(message: str) -> bool:
    """
    Detect implicit conclusion or topic switch signal.
    User is changing topic or moving on without explicit confirmation.
    
    Returns: True if implicit conclusion signal detected
    """
    message_lower = message.lower().strip()
    
    for keyword in IMPLICIT_CONCLUSION_KEYWORDS:
        if keyword in message_lower:
            return True
    
    return False


def should_switch_topic(new_topic: str, current_topic: str) -> bool:
    """
    Determine if topic should switch based on new message.
    
    Returns: True if topic changed and should handle old topic
    """
    if new_topic != current_topic and current_topic != "general":
        return True
    return False


def is_new_topic_signal(message: str) -> bool:
    """
    Detect if user is starting a completely new topic conversation.
    
    Returns: True if new topic signal detected
    """
    message_lower = message.lower().strip()
    
    new_topic_signals = [
        "new trip", "new trip", "start over", "fresh start",
        "different destination", "completely new", "other trip",
    ]
    
    for signal in new_topic_signals:
        if signal in message_lower:
            return True
    
    return False


def classify_query_type(message: str) -> str:
    """
    Classify if query references past or is a new inquiry.

    Returns: "past" if references past, "new" if new inquiry
    """
    message_lower = message.lower()

    past_reference_words = [
        "earlier", "before", "previous", "last time", "that one",
        "the hotel", "the flight", "the restaurant", "we found", "we decided", "we talked",
        "remind me", "what did we", "do you remember", "mentioned",
        "earlier we", "before we", "as i said", "as we discussed",
        "the one we", "that was", "those were", "what we chose",
        "our decision", "our choice", "agreed on", "settled on",
        "what was that", "show me the", "give me the", "list the",
        "repeat", "again", "same", "still", "already",
        "the same hotel", "the same flight", "the one you",
    ]

    for word in past_reference_words:
        if word in message_lower:
            return "past"

    return "new"