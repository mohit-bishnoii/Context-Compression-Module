import os
from dotenv import load_dotenv
from travel_agent.tools import weather_fetch, web_search, places_search
from ccm.ccm_core import ContextCompressionModule

load_dotenv()

print("Testing Weather API...")
try:
    weather = weather_fetch("London")
    print(weather)
except Exception as e:
    print("Weather API Error:", e)

print("\nTesting Web Search (SerpAPI)...")
try:
    search = web_search("flight to london")
    print(search)
except Exception as e:
    print("Web Search Error:", e)

print("\nTesting Places API (Geoapify)...")
try:
    places = places_search("London", "hotels")
    print(places)
except Exception as e:
    print("Places API Error:", e)
    
print("\nTesting Groq API (CCM Extractor)...")
try:
    ccm = ContextCompressionModule()
    res = ccm.process_user_message("I want to travel to London next week.")
    print("Extracted facts:", res)
except Exception as e:
    print("Groq API Error:", e)
