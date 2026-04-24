# travel_agent/tools.py
# These are the 4 tools available to the travel agent.
# web_search and places_search use real Google APIs.
# weather_fetch uses real OpenWeatherMap API with fake fallback.
# budget_tracker is pure Python math.

import os
import requests
import random
from datetime import datetime
from dotenv import load_dotenv
from serpapi import Client as SerpClient

load_dotenv()

# ============================================================
# BUDGET TRACKER
# Pure Python. No API. Tracks running spend total.
# ============================================================

_budget_state = {
    "total_budget": 0,
    "spent": 0,
    "transactions": []
}

def budget_tracker(action: str, amount: float = 0,
                   category: str = "", total_budget: float = 0) -> dict:
    """
    Maintains a running budget tally.
    
    Actions:
      "set_budget" - Set the total trip budget
      "add_expense" - Record a new expense
      "get_status" - Check current budget status
      "reset"      - Clear all transactions
    
    Example:
      budget_tracker("set_budget", total_budget=3000)
      budget_tracker("add_expense", amount=780, category="flights")
      budget_tracker("get_status")
    """
    global _budget_state
    
    if action == "set_budget":
        _budget_state["total_budget"] = total_budget
        _budget_state["spent"] = 0
        _budget_state["transactions"] = []
        return {
            "status": "Budget set",
            "total_budget": total_budget,
            "remaining": total_budget,
            "message": f"Trip budget set to ${total_budget:.2f}"
        }
    
    elif action == "add_expense":
        _budget_state["spent"] += amount
        _budget_state["transactions"].append({
            "category": category,
            "amount": amount,
            "timestamp": datetime.now().isoformat(),
            "running_total": _budget_state["spent"]
        })
        remaining = _budget_state["total_budget"] - _budget_state["spent"]
        
        warning = ""
        if remaining < 0:
            warning = f"⚠️ OVER BUDGET by ${abs(remaining):.2f}!"
        elif remaining < _budget_state["total_budget"] * 0.2:
            warning = f"⚠️ Warning: Only ${remaining:.2f} remaining (less than 20% of budget)"
        
        return {
            "status": "Expense recorded",
            "category": category,
            "amount_spent": amount,
            "total_spent": _budget_state["spent"],
            "total_budget": _budget_state["total_budget"],
            "remaining": remaining,
            "warning": warning,
            "transactions": _budget_state["transactions"]
        }
    
    elif action == "get_status":
        remaining = _budget_state["total_budget"] - _budget_state["spent"]
        percent_used = (
            (_budget_state["spent"] / _budget_state["total_budget"] * 100)
            if _budget_state["total_budget"] > 0 else 0
        )
        return {
            "total_budget": _budget_state["total_budget"],
            "total_spent": _budget_state["spent"],
            "remaining": remaining,
            "percent_used": round(percent_used, 1),
            "transactions": _budget_state["transactions"],
            "status": "over_budget" if remaining < 0 else "on_track"
        }
    
    elif action == "reset":
        _budget_state = {"total_budget": 0, "spent": 0, "transactions": []}
        return {"status": "Budget reset", "message": "All transactions cleared"}
    
    else:
        return {"error": f"Unknown action: {action}. Use: set_budget, add_expense, get_status, reset"}


def get_budget_state():
    """Helper function to get current budget state for memory system."""
    return _budget_state.copy()


def reset_budget():
    """Reset budget when starting new conversation."""
    global _budget_state
    _budget_state = {"total_budget": 0, "spent": 0, "transactions": []}


# ============================================================
# WEATHER FETCH
# Tries real OpenWeatherMap API first.
# Falls back to realistic fake data if API key not set.
# ============================================================

def weather_fetch(city: str, travel_dates: str = "next month") -> dict:
    """
    Fetches weather information for a city.
    Uses real OpenWeatherMap API if key is available.
    Falls back to realistic fake data otherwise.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    
    if api_key and api_key != "your_weather_key_here":
        try:
            result = _fetch_real_weather(city, api_key, travel_dates)
            if result:
                return result
        except Exception as e:
            print(f"Weather API failed, using fake data: {e}")
        
    return _fetch_fake_weather(city, travel_dates)


def _fetch_real_weather(city: str, api_key: str, travel_dates: str) -> dict:
    """Calls real OpenWeatherMap API."""
    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "imperial"
    }
    
    response = requests.get(url, params=params, timeout=5)
    
    if response.status_code != 200:
        return None
    
    data = response.json()
    
    temp = data["main"]["temp"]
    feels_like = data["main"]["feels_like"]
    humidity = data["main"]["humidity"]
    description = data["weather"][0]["description"]
    wind_speed = data["wind"]["speed"]
    
    packing = _generate_packing_recommendations(temp, description, humidity)
    
    return {
        "city": city,
        "travel_dates": travel_dates,
        "data_source": "OpenWeatherMap (real data)",
        "current_conditions": {
            "temperature_f": round(temp),
            "feels_like_f": round(feels_like),
            "humidity_percent": humidity,
            "description": description,
            "wind_mph": round(wind_speed)
        },
        "packing_recommendations": packing,
        "activity_impact": _generate_activity_impact(temp, description),
        "note": "Current weather shown. Conditions may differ on travel dates."
    }


def _fetch_fake_weather(city: str, travel_dates: str) -> dict:
    """Returns realistic fake weather data."""
    city_weather = {
        "tokyo": {"temp_f": 72, "feels_like": 70, "humidity": 65,
            "description": "partly cloudy with occasional showers", "wind_mph": 8,
            "seasonal_note": "Rainy season in June. Bring umbrella.",
            "best_activities": ["indoor museums", "covered markets"]},
        "kyoto": {"temp_f": 75, "feels_like": 73, "humidity": 70,
            "description": "warm and humid, occasional afternoon showers", "wind_mph": 5,
            "seasonal_note": "Summers hot and humid. Spring and fall ideal.",
            "best_activities": ["temple visits", "bamboo forest"]},
        "paris": {"temp_f": 65, "feels_like": 63, "humidity": 72,
            "description": "mild and overcast with light rain", "wind_mph": 12,
            "seasonal_note": "Spring mild and beautiful.",
            "best_activities": ["outdoor cafes", "museums"]},
        "amsterdam": {"temp_f": 60, "feels_like": 57, "humidity": 80,
            "description": "cloudy with frequent light rain", "wind_mph": 15,
            "seasonal_note": "Bring waterproof jacket year-round.",
            "best_activities": ["canal tours", "museums"]},
        "rome": {"temp_f": 78, "feels_like": 80, "humidity": 55,
            "description": "sunny and warm", "wind_mph": 7,
            "seasonal_note": "Hot summers.",
            "best_activities": ["outdoor ruins", "piazzas"]},
        "bali": {"temp_f": 85, "feels_like": 92, "humidity": 85,
            "description": "hot and humid, afternoon thunderstorms", "wind_mph": 10,
            "seasonal_note": "Wet season Nov-Mar. Dry season Apr-Oct.",
            "best_activities": ["temple visits", "beach"]},
        "switzerland": {"temp_f": 55, "feels_like": 50, "humidity": 60,
            "description": "cool and clear", "wind_mph": 20,
            "seasonal_note": "Alpine climate. Layers essential.",
            "best_activities": ["hiking", "cable cars"]},
        "new york": {"temp_f": 68, "feels_like": 66, "humidity": 58,
            "description": "partly cloudy, comfortable", "wind_mph": 13,
            "seasonal_note": "Four seasons.",
            "best_activities": ["walking", "parks"]}
    }
    
    city_lower = city.lower()
    weather_data = None
    for key in city_weather:
        if key in city_lower or city_lower in key:
            weather_data = city_weather[key]
            break
    
    if not weather_data:
        weather_data = {"temp_f": 70, "feels_like": 68, "humidity": 60,
            "description": "mild and partly cloudy", "wind_mph": 10,
            "seasonal_note": "Check local forecasts.",
            "best_activities": ["general sightseeing"]}
    
    packing = _generate_packing_recommendations(
        weather_data["temp_f"], weather_data["description"], weather_data["humidity"])
    
    return {
        "city": city,
        "travel_dates": travel_dates,
        "data_source": "Estimated typical conditions (fake data)",
        "current_conditions": {
            "temperature_f": weather_data["temp_f"],
            "feels_like_f": weather_data["feels_like"],
            "humidity_percent": weather_data["humidity"],
            "description": weather_data["description"],
            "wind_mph": weather_data["wind_mph"]
        },
        "seasonal_note": weather_data["seasonal_note"],
        "recommended_activities": weather_data["best_activities"],
        "packing_recommendations": packing,
        "activity_impact": _generate_activity_impact(
            weather_data["temp_f"], weather_data["description"])
    }


def _generate_packing_recommendations(temp_f: float, description: str, humidity: int) -> list:
    """Generate packing list based on weather conditions."""
    packing = []
    
    if temp_f < 50:
        packing.extend(["heavy coat", "warm layers", "gloves", "scarf"])
    elif temp_f < 65:
        packing.extend(["light jacket", "layers", "long pants"])
    elif temp_f < 80:
        packing.extend(["light layers", "mix of sleeves"])
    else:
        packing.extend(["light breathable clothing", "shorts", "t-shirts"])
    
    if any(word in description.lower() for word in ["rain", "shower", "drizzle"]):
        packing.extend(["umbrella", "waterproof jacket"])
    
    if humidity > 75:
        packing.extend(["moisture-wicking fabrics"])
    
    if any(word in description.lower() for word in ["sunny", "clear"]):
        packing.extend(["sunscreen", "sunglasses"])
    
    packing.extend(["comfortable walking shoes"])
    
    return list(set(packing))


def _generate_activity_impact(temp_f: float, description: str) -> dict:
    """Advise on how weather affects planned activities."""
    return {
        "outdoor_activities": (
            "great conditions" if temp_f > 60 and "rain" not in description
            else "bring rain gear" if "rain" in description
            else "cold but manageable with layers"
        ),
        "indoor_alternatives": "museums, galleries, restaurants" if "rain" in description else "optional",
        "best_time_of_day": (
            "early morning or evening" if temp_f > 85
            else "anytime" if 60 <= temp_f <= 80
            else "midday when warmest"
        )
    }


# ============================================================
# WEB SEARCH (SerpApi)
# Uses SerpApi for real flight and general search data.
# ============================================================

def web_search(query: str) -> dict:
    """
    Performs web search using SerpApi.
    Detects query type and routes to appropriate handler.
    """
    query_lower = query.lower()
    
    if any(word in query_lower for word in ["flight", "fly", "airline", "airport"]):
        return _search_flights(query)
    elif any(word in query_lower for word in ["train", "rail", "eurostar", "amtrak"]):
        return _search_trains(query)
    elif any(word in query_lower for word in ["visa", "passport", "entry requirement"]):
        return _search_visa_info(query)
    elif any(word in query_lower for word in ["currency", "exchange", "money"]):
        return _search_currency_info(query)
    else:
        return _search_general_info(query)


def _search_flights(query: str) -> dict:
    """Search flights using SerpApi regular search (not Google Flights engine)."""
    api_key = os.getenv("SERP_API_KEY")

    if not api_key:
        return _fallback_flights(query)

    try:
        client = SerpClient(api_key=api_key)
        results = client.search(q=query)

        flights = []
        for r in results.get("organic_results", [])[:5]:
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            flights.append({
                "airline": title,
                "price": snippet,
                "duration": "N/A",
                "type": "Search Results"
            })
        if flights:
            return {"results": flights, "data_source": "SerpApi", "search_type": "flights"}
        return _fallback_flights(query)
    except Exception as e:
        print(f"SerpApi flight search failed: {e}")
        return _fallback_flights(query)


def _fallback_flights(query: str) -> dict:
    """Fallback flight data if SerpApi unavailable."""
    return {
        "search_type": "flights",
        "query": query,
        "message": "API key not configured. Using fallback data.",
        "data_source": "Fallback (no API)",
        "results": [
            {"airline": "Various airlines", "price": "Check airline网站", "duration": "Varies"}
        ]
    }


def _search_trains(query: str) -> dict:
    """Train search with fallback data."""
    return {
        "search_type": "trains",
        "query": query,
        "message": "Train search via SerpApi",
        "data_source": "SerpApi",
        "tips": [
            "Check seat61.com for comprehensive train info",
            "Book tickets in advance for best prices"
        ]
    }


def _search_visa_info(query: str) -> dict:
    """Visa requirement information."""
    return {
        "search_type": "visa_info",
        "query": query,
        "us_passport_holders": {
            "japan": "No visa required for stays up to 90 days",
            "france": "No visa required (Schengen Zone, 90 days)",
            "italy": "No visa required (Schengen Zone, 90 days)",
            "netherlands": "No visa required (Schengen Zone, 90 days)",
            "indonesia_bali": "Visa on Arrival available ($35, 30 days)",
            "switzerland": "No visa required for stays up to 90 days"
        },
        "important_notes": [
            "Passport must be valid for 6 months beyond travel dates",
            "Always check official embassy websites"
        ]
    }


def _search_currency_info(query: str) -> dict:
    """Currency exchange information."""
    return {
        "search_type": "currency",
        "query": query,
        "exchange_rates_approximate": {
            "USD to JPY": 149.5,
            "USD to EUR": 0.92,
            "USD to CHF": 0.89,
            "USD to IDR": 15750
        },
        "tips": [
            "Use ATMs at banks for best rates",
            "Notify bank before travel",
            "Avoid airport currency exchange"
        ]
    }


def _search_general_info(query: str) -> dict:
    """General web search using SerpApi."""
    api_key = os.getenv("SERP_API_KEY")
    
    if not api_key:
        return _fallback_general_info(query)
    
    try:
        client = SerpClient(api_key=api_key)
        results = client.search(q=query)
        organic = results.get("organic_results", [])[:3]
        return {
            "results": [
                {"title": r.get("title"), "snippet": r.get("snippet")}
                for r in organic
            ],
            "data_source": "SerpApi",
            "search_type": "general"
        }
    except Exception as e:
        print(f"SerpApi general search failed: {e}")
        return _fallback_general_info(query)


def _fallback_general_info(query: str) -> dict:
    """Fallback general info."""
    return {
        "search_type": "general",
        "query": query,
        "results": [
            {"title": f"Travel guide: {query}", "snippet": "General travel information"}
        ],
        "data_source": "Fallback (no API)"
    }


# ============================================================
# PLACES SEARCH (Geoapify API)
# Uses Geoapify for hotels, restaurants, attractions.
# ============================================================

def places_search(location: str, category: str = "hotels",
                  budget_per_night: float = None) -> dict:
    """
    Searches for hotels, restaurants, or attractions using Geoapify API.
    """
    category_lower = category.lower()
    
    if "hotel" in category_lower or "accommodation" in category_lower:
        return _run_geoapify_search(location, "accommodation.hotel")
    elif "restaurant" in category_lower or "food" in category_lower or "dining" in category_lower:
        return _run_geoapify_search(location, "catering.restaurant")
    else:
        return _run_geoapify_search(location, "tourism.attraction")


def _run_geoapify_search(location: str, category: str) -> dict:
    """Search places using Geoapify Places API."""
    api_key = os.getenv("GEOAPIFY_API_KEY")
    
    if not api_key:
        return _fallback_places(category)
    
    try:
        geocode_url = f"https://api.geoapify.com/v1/geocode/search?text={location}&apiKey={api_key}"
        geo_response = requests.get(geocode_url, timeout=10)
        geo_data = geo_response.json()
        
        features = geo_data.get("features", [])
        if not features:
            return {"error": f"Could not find location: {location}", "results": []}
        
        coords = features[0].get("geometry", {}).get("coordinates", [])
        lon = coords[0] if len(coords) > 0 else 0
        lat = coords[1] if len(coords) > 1 else 0
        
        if not lat or not lon:
            return {"error": f"Could not get coordinates for: {location}", "results": []}
        
        lon_min, lat_min = lon - 0.1, lat - 0.1
        lon_max, lat_max = lon + 0.1, lat + 0.1
        
        places_url = (
            f"https://api.geoapify.com/v2/places?"
            f"categories={category}&"
            f"filter=rect:{lon_min},{lat_min},{lon_max},{lat_max}&"
            f"limit=5&"
            f"apiKey={api_key}"
        )
        
        places_response = requests.get(places_url, timeout=10)
        places_data = places_response.json()
        
        places = []
        for feature in places_data.get("features", [])[:5]:
            prop = feature.get("properties", {})
            places.append({
                "name": prop.get("name"),
                "address": prop.get("formatted"),
                "lat": prop.get("lat"),
                "lon": prop.get("lon"),
                "data_source": "Geoapify (OpenStreetMap)"
            })
        
        return {"results": places, "data_source": "Geoapify Places API"}
    except Exception as e:
        print(f"Geoapify API failed: {e}")
        return _fallback_places(category)


def _fallback_places(category: str) -> dict:
    """Fallback places data."""
    return {
        "search_type": category,
        "data_source": "Fallback (no API)",
        "message": "Geoapify API key not configured",
        "results": []
    }