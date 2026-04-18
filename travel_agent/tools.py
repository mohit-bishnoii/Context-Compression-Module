# travel_agent/tools.py
# These are the 4 tools available to the travel agent.
# web_search and places_search use realistic fake data.
# weather_fetch uses real OpenWeatherMap API with fake fallback.
# budget_tracker is pure Python math.

import os
import requests
import random
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# BUDGET TRACKER
# Pure Python. No API. Tracks running spend total.
# ============================================================

# This dict lives in memory while the program runs.
# It gets reset when you start a new conversation.
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
        
        # Flag if over budget
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
    
    Example:
      weather_fetch("Tokyo", "June 3-13")
      weather_fetch("Paris", "next week")
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    
    # Try real API first
    if api_key and api_key != "your_weather_key_here":
        try:
            result = _fetch_real_weather(city, api_key, travel_dates)
            if result:
                return result
        except Exception as e:
            print(f"Weather API failed, using fake data: {e}")
        
    # Fall back to realistic fake data
    return _fetch_fake_weather(city, travel_dates)


def _fetch_real_weather(city: str, api_key: str, travel_dates: str) -> dict:
    """Calls real OpenWeatherMap API."""
    
    # Get current weather (free API)
    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "imperial"  # Fahrenheit
    }
    
    response = requests.get(url, params=params, timeout=5)
    
    if response.status_code != 200:
        return None
    
    data = response.json()
    
    # Extract useful info from real API response
    temp = data["main"]["temp"]
    feels_like = data["main"]["feels_like"]
    humidity = data["main"]["humidity"]
    description = data["weather"][0]["description"]
    wind_speed = data["wind"]["speed"]
    
    # Generate packing recommendations based on real data
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
    """
    Returns realistic fake weather data.
    Data is based on real typical weather for major cities.
    """
    
    # Realistic weather profiles for common destinations
    city_weather = {
        "tokyo": {
            "temp_f": 72, "feels_like": 70, "humidity": 65,
            "description": "partly cloudy with occasional showers",
            "wind_mph": 8,
            "seasonal_note": "Rainy season in June. Bring umbrella. Cherry blossoms in April.",
            "best_activities": ["indoor museums", "covered markets", "tea ceremonies"]
        },
        "kyoto": {
            "temp_f": 75, "feels_like": 73, "humidity": 70,
            "description": "warm and humid, occasional afternoon showers",
            "wind_mph": 5,
            "seasonal_note": "Traditional Japan. Summers hot and humid. Spring and fall ideal.",
            "best_activities": ["temple visits (early morning)", "bamboo forest", "geisha district"]
        },
        "paris": {
            "temp_f": 65, "feels_like": 63, "humidity": 72,
            "description": "mild and overcast with light rain",
            "wind_mph": 12,
            "seasonal_note": "Spring mild and beautiful. Summer busy but warm. Always pack layers.",
            "best_activities": ["outdoor cafes", "museums", "river walks", "markets"]
        },
        "amsterdam": {
            "temp_f": 60, "feels_like": 57, "humidity": 80,
            "description": "cloudy with frequent light rain",
            "wind_mph": 15,
            "seasonal_note": "Bring waterproof jacket year-round. Tulip season April-May.",
            "best_activities": ["canal tours", "museums", "cycling when dry"]
        },
        "rome": {
            "temp_f": 78, "feels_like": 80, "humidity": 55,
            "description": "sunny and warm",
            "wind_mph": 7,
            "seasonal_note": "Hot summers. Visit monuments early morning. Fountains everywhere.",
            "best_activities": ["outdoor ruins", "piazzas", "food markets", "walking tours"]
        },
        "bali": {
            "temp_f": 85, "feels_like": 92, "humidity": 85,
            "description": "hot and humid, afternoon thunderstorms",
            "wind_mph": 10,
            "seasonal_note": "Wet season Nov-Mar. Dry season Apr-Oct best for travel.",
            "best_activities": ["morning temple visits", "beach before noon", "rice terrace walks"]
        },
        "switzerland": {
            "temp_f": 55, "feels_like": 50, "humidity": 60,
            "description": "cool and clear, excellent mountain visibility",
            "wind_mph": 20,
            "seasonal_note": "Alpine climate. Layers essential. Skiing Dec-Mar. Hiking Jun-Sep.",
            "best_activities": ["hiking", "cable cars", "lake walks", "skiing in winter"]
        },
        "new york": {
            "temp_f": 68, "feels_like": 66, "humidity": 58,
            "description": "partly cloudy, comfortable",
            "wind_mph": 13,
            "seasonal_note": "Four seasons. Summer hot, winter cold. Fall and spring ideal.",
            "best_activities": ["walking", "parks", "outdoor markets", "neighborhoods"]
        }
    }
    
    # Look up city (case insensitive, partial match)
    city_lower = city.lower()
    weather_data = None
    for key in city_weather:
        if key in city_lower or city_lower in key:
            weather_data = city_weather[key]
            break
    
    # Default weather if city not in our list
    if not weather_data:
        weather_data = {
            "temp_f": 70, "feels_like": 68, "humidity": 60,
            "description": "mild and partly cloudy",
            "wind_mph": 10,
            "seasonal_note": "Check local forecasts closer to travel date.",
            "best_activities": ["general sightseeing", "outdoor activities when dry"]
        }
    
    packing = _generate_packing_recommendations(
        weather_data["temp_f"],
        weather_data["description"],
        weather_data["humidity"]
    )
    
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
            weather_data["temp_f"],
            weather_data["description"]
        )
    }


def _generate_packing_recommendations(temp_f: float,
                                       description: str, humidity: int) -> list:
    """Generate packing list based on weather conditions."""
    packing = []
    
    # Temperature-based
    if temp_f < 50:
        packing.extend(["heavy coat", "warm layers", "gloves", "scarf", "thermal underwear"])
    elif temp_f < 65:
        packing.extend(["light jacket", "layers", "long pants", "light sweater"])
    elif temp_f < 80:
        packing.extend(["light layers", "mix of short and long sleeves", "light cardigan"])
    else:
        packing.extend(["light breathable clothing", "shorts", "t-shirts", "sun hat"])
    
    # Rain-based
    if any(word in description.lower() for word in ["rain", "shower", "drizzle", "wet"]):
        packing.extend(["umbrella", "waterproof jacket", "waterproof shoes"])
    
    # Humidity-based
    if humidity > 75:
        packing.extend(["moisture-wicking fabrics", "extra clothing changes"])
    
    # Sun-based
    if any(word in description.lower() for word in ["sunny", "clear", "hot"]):
        packing.extend(["sunscreen SPF 50", "sunglasses", "sun hat"])
    
    # Always include
    packing.extend(["comfortable walking shoes", "day bag/backpack"])
    
    return list(set(packing))  # Remove duplicates


def _generate_activity_impact(temp_f: float, description: str) -> dict:
    """Advise on how weather affects planned activities."""
    return {
        "outdoor_activities": (
            "great conditions" if temp_f > 60 and "rain" not in description
            else "bring rain gear" if "rain" in description
            else "cold but manageable with layers"
        ),
        "indoor_alternatives": "museums, galleries, restaurants, shopping" if "rain" in description else "optional",
        "best_time_of_day": (
            "early morning or evening" if temp_f > 85
            else "anytime" if 60 <= temp_f <= 80
            else "midday when warmest"
        )
    }


# ============================================================
# WEB SEARCH (FAKE)
# Returns realistic flight and general info data.
# ============================================================

def web_search(query: str) -> dict:
    """
    Simulates a web search for flights and general travel information.
    Returns realistic fake data with enough variety to be useful.
    
    Example:
      web_search("flights from New York to Tokyo June")
      web_search("best time to visit Kyoto")
      web_search("train from Paris to Amsterdam")
    """
    query_lower = query.lower()
    
    # Detect what kind of search this is
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
    """Generate realistic flight search results."""
    query_lower = query.lower()
    
    # Detect destination
    destinations = {
        "tokyo": ("JFK", "NRT", "Tokyo Narita", 14, [780, 850, 920, 760, 990]),
        "kyoto": ("JFK", "KIX", "Osaka Kansai", 14, [820, 900, 780, 950]),
        "paris": ("JFK", "CDG", "Paris Charles de Gaulle", 7, [450, 520, 480, 600, 430]),
        "amsterdam": ("JFK", "AMS", "Amsterdam Schiphol", 8, [480, 550, 510, 590]),
        "rome": ("JFK", "FCO", "Rome Fiumicino", 9, [520, 480, 600, 560]),
        "bali": ("JFK", "DPS", "Bali Ngurah Rai", 22, [950, 1100, 1050, 880]),
        "switzerland": ("JFK", "ZRH", "Zurich", 9, [600, 680, 720, 650]),
    }
    
    dest_key = None
    dest_info = None
    for key, info in destinations.items():
        if key in query_lower:
            dest_key = key
            dest_info = info
            break
    
    if not dest_info:
        # Generic flight results
        dest_info = ("JFK", "DEST", "Destination Airport", 10, [500, 600, 700])
        dest_key = "destination"
    
    origin, dest_code, dest_airport, flight_hours, prices = dest_info
    
    airlines = [
        ("ANA", "All Nippon Airways", "NH"),
        ("JAL", "Japan Airlines", "JL"),
        ("United", "United Airlines", "UA"),
        ("Delta", "Delta Air Lines", "DL"),
        ("Korean Air", "Korean Air", "KE"),
        ("Air France", "Air France", "AF"),
        ("Lufthansa", "Lufthansa", "LH"),
        ("Emirates", "Emirates", "EK")
    ]
    
    # Generate 5 flight options
    results = []
    for i, price in enumerate(prices[:5]):
        airline = airlines[i % len(airlines)]
        stops = 0 if i % 3 == 0 else 1
        duration = flight_hours if stops == 0 else flight_hours + 2
        
        results.append({
            "airline": airline[0],
            "airline_full": airline[1],
            "flight_number": f"{airline[2]}-{random.randint(100, 999)}",
            "price_usd": price,
            "class": "Economy",
            "stops": stops,
            "stop_city": "Seoul" if stops == 1 else None,
            "duration_hours": duration,
            "departure": f"{origin} 11:30 PM",
            "arrival": f"{dest_code} 5:30 AM (+{2 if stops==0 else 3} days)",
            "baggage_included": i % 2 == 0,
            "meal_included": True,
            "refundable": i % 3 == 0,
            "seats_left": random.randint(3, 15)
        })
    
    # Sort by price
    results.sort(key=lambda x: x["price_usd"])
    
    return {
        "search_type": "flights",
        "query": query,
        "route": f"{origin} → {dest_code}",
        "results_count": len(results),
        "cheapest_price": results[0]["price_usd"],
        "results": results,
        "booking_tips": [
            "Book 6-8 weeks in advance for best prices",
            "Tuesday and Wednesday departures often cheaper",
            "Check baggage fees — some cheap fares add $50-100 in fees",
            f"Flight time approximately {flight_hours} hours nonstop"
        ],
        "data_note": "Prices are estimates. Check airline websites for actual fares."
    }


def _search_trains(query: str) -> dict:
    """Generate realistic train search results."""
    query_lower = query.lower()
    
    train_routes = {
        ("paris", "amsterdam"): {
            "operator": "Thalys / Eurostar",
            "duration": "3h 20min",
            "distance_km": 514,
            "prices": [79, 99, 129, 189, 249],
            "frequency": "8 trains per day",
            "departure_station": "Paris Gare du Nord",
            "arrival_station": "Amsterdam Centraal",
            "departures": ["06:13", "07:13", "09:13", "11:13", "13:13",
                          "15:13", "17:13", "19:13"]
        },
        ("paris", "rome"): {
            "operator": "Trenitalia / SNCF",
            "duration": "11h 00min",
            "distance_km": 1421,
            "prices": [89, 129, 189],
            "frequency": "1-2 overnight trains per day",
            "departure_station": "Paris Gare de Lyon",
            "arrival_station": "Roma Termini",
            "departures": ["19:00", "21:00"]
        },
        ("tokyo", "kyoto"): {
            "operator": "JR Shinkansen (Bullet Train)",
            "duration": "2h 15min",
            "distance_km": 513,
            "prices": [68, 85, 120],
            "frequency": "Every 10 minutes",
            "departure_station": "Tokyo Station",
            "arrival_station": "Kyoto Station",
            "departures": ["Every 10 minutes from 6:00 AM to 10:00 PM"]
        }
    }
    
    # Find matching route
    route_data = None
    for (origin, dest), data in train_routes.items():
        if origin in query_lower and dest in query_lower:
            route_data = data
            break
        if dest in query_lower and origin in query_lower:
            route_data = data
            break
    
    if not route_data:
        return {
            "search_type": "trains",
            "query": query,
            "message": "Specific route not found in database",
            "general_tips": [
                "Book train tickets in advance for best prices",
                "Rail passes available for multi-city European trips",
                "Check seat61.com for comprehensive train travel info"
            ]
        }
    
    return {
        "search_type": "trains",
        "query": query,
        "operator": route_data["operator"],
        "duration": route_data["duration"],
        "frequency": route_data["frequency"],
        "departure_station": route_data["departure_station"],
        "arrival_station": route_data["arrival_station"],
        "departure_times": route_data["departures"],
        "price_range_usd": {
            "economy": route_data["prices"][0],
            "standard": route_data["prices"][1] if len(route_data["prices"]) > 1 else None,
            "business": route_data["prices"][-1]
        },
        "booking_tips": [
            "Book online in advance for cheapest fares",
            "Bring passport for international trains",
            f"Travel time: {route_data['duration']}"
        ]
    }


def _search_visa_info(query: str) -> dict:
    """Return visa requirement information."""
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
            "Always check official embassy websites for current requirements",
            "Travel insurance strongly recommended"
        ]
    }


def _search_currency_info(query: str) -> dict:
    """Return currency exchange information."""
    return {
        "search_type": "currency",
        "query": query,
        "exchange_rates_approximate": {
            "USD to JPY (Japan)": 149.5,
            "USD to EUR (Europe)": 0.92,
            "USD to CHF (Switzerland)": 0.89,
            "USD to IDR (Bali)": 15750
        },
        "tips": [
            "Use ATMs at banks for best exchange rates",
            "Notify your bank before international travel",
            "Avoid airport currency exchange (worst rates)",
            "Credit cards with no foreign transaction fees save 3%",
            "Japan is still largely cash-based — carry JPY"
        ]
    }


def _search_general_info(query: str) -> dict:
    """Return general travel information."""
    return {
        "search_type": "general",
        "query": query,
        "results": [
            {
                "title": f"Travel guide: {query}",
                "summary": (
                    "Based on traveler reviews and official tourism information, "
                    "this destination offers excellent experiences for most travel "
                    "styles. Best visited with advance planning and bookings."
                ),
                "source": "Travel Guide Database",
                "relevance": "high"
            },
            {
                "title": "Tips from recent travelers",
                "summary": (
                    "Recent visitors recommend booking popular attractions in "
                    "advance, especially during peak season. Local transportation "
                    "is generally reliable and affordable."
                ),
                "source": "Traveler Reviews",
                "relevance": "medium"
            }
        ],
        "note": "General information based on typical travel patterns"
    }


# ============================================================
# PLACES SEARCH (FAKE)
# Returns hotels, restaurants, attractions with rich data.
# ============================================================

def places_search(location: str, category: str = "hotels",
                  budget_per_night: float = None) -> dict:
    """
    Searches for hotels, restaurants, or attractions in a location.
    Returns realistic fake data with enough variety to be useful.
    
    Example:
      places_search("Tokyo", "hotels", budget_per_night=150)
      places_search("Tsukiji Tokyo", "restaurants")
      places_search("Kyoto", "attractions")
    """
    category_lower = category.lower()
    
    if "hotel" in category_lower or "accommodation" in category_lower:
        return _search_hotels(location, budget_per_night)
    elif "restaurant" in category_lower or "food" in category_lower or "dining" in category_lower:
        return _search_restaurants(location)
    elif "attraction" in category_lower or "activity" in category_lower or "thing" in category_lower:
        return _search_attractions(location)
    else:
        return _search_hotels(location, budget_per_night)


def _search_hotels(location: str, budget: float = None) -> dict:
    """Generate realistic hotel search results."""
    location_lower = location.lower()
    
    hotel_database = {
        "tokyo": [
            {"name": "Park Hyatt Tokyo", "area": "Shinjuku", "price_per_night": 380,
             "rating": 4.8, "stars": 5, "style": "luxury",
             "features": ["Sky Bar", "Pool", "City views", "Fine dining"],
             "distance_to_center": "0.5km from Shinjuku station",
             "dietary_options": ["extensive", "allergy-aware menus available"]},
            {"name": "Shinjuku Granbell Hotel", "area": "Shinjuku", "price_per_night": 145,
             "rating": 4.3, "stars": 4, "style": "boutique",
             "features": ["Modern design", "Great location", "24h front desk"],
             "distance_to_center": "3 min walk to Shinjuku station",
             "dietary_options": ["standard menu", "can accommodate requests"]},
            {"name": "Richmond Hotel Premier Tokyo", "area": "Asakusa", "price_per_night": 120,
             "rating": 4.2, "stars": 3, "style": "business",
             "features": ["Traditional area", "Near Senso-ji temple", "Clean rooms"],
             "distance_to_center": "5 min walk to Asakusa station",
             "dietary_options": ["Japanese breakfast available"]},
            {"name": "Khaosan Tokyo Kabuki", "area": "Asakusa", "price_per_night": 45,
             "rating": 4.0, "stars": 2, "style": "hostel/budget",
             "features": ["Social atmosphere", "Central location", "Free WiFi"],
             "distance_to_center": "Near Asakusa",
             "dietary_options": ["vending machines only"]},
            {"name": "Cerulean Tower Tokyu Hotel", "area": "Shibuya", "price_per_night": 280,
             "rating": 4.6, "stars": 5, "style": "upscale",
             "features": ["Mountain views", "Jazz bar", "Multiple restaurants"],
             "distance_to_center": "Shibuya crossing walkable",
             "dietary_options": ["multiple restaurants", "allergy requests honored"]}
        ],
        "kyoto": [
            {"name": "Tawaraya Ryokan", "area": "Central Kyoto", "price_per_night": 500,
             "rating": 4.9, "stars": 5, "style": "traditional ryokan",
             "features": ["400-year-old inn", "Kaiseki meals included", "Tea ceremony"],
             "distance_to_center": "Central location",
             "dietary_options": ["traditional kaiseki", "dietary needs accommodated with notice"]},
            {"name": "The Mitsui Kyoto", "area": "Nijo Castle area", "price_per_night": 420,
             "rating": 4.7, "stars": 5, "style": "luxury",
             "features": ["Historic building", "Garden", "Spa", "Multiple restaurants"],
             "distance_to_center": "10 min from downtown",
             "dietary_options": ["extensive", "allergy-aware"]},
            {"name": "Kyoto Tower Hotel", "area": "Kyoto Station", "price_per_night": 130,
             "rating": 4.1, "stars": 3, "style": "business",
             "features": ["Station adjacent", "Convenient location", "City views"],
             "distance_to_center": "At Kyoto Station",
             "dietary_options": ["standard hotel breakfast"]},
            {"name": "Guest House Waraku-An", "area": "Gion", "price_per_night": 85,
             "rating": 4.4, "stars": 2, "style": "guesthouse",
             "features": ["Traditional Machiya house", "Gion district", "Authentic experience"],
             "distance_to_center": "Geisha district",
             "dietary_options": ["self-catering kitchen available"]}
        ],
        "paris": [
            {"name": "Le Meurice", "area": "1st arrondissement", "price_per_night": 850,
             "rating": 4.9, "stars": 5, "style": "palace hotel",
             "features": ["Tuileries Garden view", "Michelin-starred restaurant", "Historic"],
             "distance_to_center": "Louvre 5 min walk",
             "dietary_options": ["extensive fine dining", "all allergies accommodated"]},
            {"name": "Hotel Fabric", "area": "11th arrondissement", "price_per_night": 180,
             "rating": 4.5, "stars": 4, "style": "boutique",
             "features": ["Converted factory", "Hip area", "Great breakfast"],
             "distance_to_center": "Oberkampf metro",
             "dietary_options": ["French breakfast", "vegan options"]},
            {"name": "Hotel des Arts Montmartre", "area": "Montmartre", "price_per_night": 120,
             "rating": 4.2, "stars": 3, "style": "charming",
             "features": ["Artist neighborhood", "Sacré-Cœur views", "Authentic Paris"],
             "distance_to_center": "Montmartre",
             "dietary_options": ["continental breakfast"]},
            {"name": "Generator Paris", "area": "10th arrondissement", "price_per_night": 55,
             "rating": 4.0, "stars": 2, "style": "hostel/budget",
             "features": ["Social hostel", "Rooftop bar", "Central location"],
             "distance_to_center": "Gare de l'Est 5 min",
             "dietary_options": ["basic cafe"]}
        ],
        "amsterdam": [
            {"name": "Waldorf Astoria Amsterdam", "area": "Canal Ring", "price_per_night": 650,
             "rating": 4.8, "stars": 5, "style": "luxury",
             "features": ["17th century canal house", "Michelin restaurant", "Spa"],
             "distance_to_center": "Herengracht canal",
             "dietary_options": ["fine dining", "all restrictions accommodated"]},
            {"name": "Hotel V Nesplein", "area": "City Center", "price_per_night": 185,
             "rating": 4.4, "stars": 4, "style": "boutique",
             "features": ["Design hotel", "Lively bar", "Central location"],
             "distance_to_center": "City center",
             "dietary_options": ["modern European menu"]},
            {"name": "Stayokay Amsterdam Stadsdoelen", "area": "City Center",
             "price_per_night": 45, "rating": 4.1, "stars": 2, "style": "hostel",
             "features": ["Historic building", "Central", "Social"],
             "distance_to_center": "Nieuwmarkt area",
             "dietary_options": ["basic meals"]}
        ]
    }
    
    # Find location in database
    location_lower = location.lower()
    hotels = None
    for key in hotel_database:
        if key in location_lower:
            hotels = hotel_database[key]
            break
    
    # Default if city not in database
    if not hotels:
        hotels = [
            {"name": f"{location} Grand Hotel", "area": "City Center",
             "price_per_night": 180, "rating": 4.3, "stars": 4, "style": "standard",
             "features": ["Central location", "Restaurant", "WiFi"],
             "distance_to_center": "City center",
             "dietary_options": ["standard menu"]},
            {"name": f"{location} Budget Inn", "area": "City Center",
             "price_per_night": 80, "rating": 3.9, "stars": 2, "style": "budget",
             "features": ["Good value", "Clean rooms"],
             "distance_to_center": "Near center",
             "dietary_options": ["vending machines"]}
        ]
    
    # Filter by budget if provided
    if budget:
        hotels_in_budget = [h for h in hotels if h["price_per_night"] <= budget]
        hotels_over_budget = [h for h in hotels if h["price_per_night"] > budget]
    else:
        hotels_in_budget = hotels
        hotels_over_budget = []
    
    return {
        "search_type": "hotels",
        "location": location,
        "budget_filter": budget,
        "results_count": len(hotels),
        "within_budget": [
            {**h, "within_budget": True} for h in hotels_in_budget
        ],
        "over_budget": [
            {**h, "within_budget": False, "over_by": h["price_per_night"] - budget}
            for h in hotels_over_budget
        ] if budget else [],
        "all_results": hotels,
        "cheapest": min(hotels, key=lambda x: x["price_per_night"]),
        "highest_rated": max(hotels, key=lambda x: x["rating"]),
        "booking_tips": [
            "Book 4-6 weeks in advance for better rates",
            "Check cancellation policy before booking",
            "Read recent reviews for latest quality information"
        ]
    }


def _search_restaurants(location: str) -> dict:
    """Generate realistic restaurant search results."""
    location_lower = location.lower()
    
    restaurant_database = {
        "tsukiji": [
            {"name": "Sushi Dai", "cuisine": "Sushi/Seafood",
             "price_range": "$$", "rating": 4.8,
             "specialty": "Omakase sushi, fresh tuna, shellfish platter",
             "allergy_warning": "SHELLFISH — primary ingredient, not suitable for shellfish allergy",
             "hours": "5:00 AM - 2:00 PM", "wait_time": "1-3 hours",
             "reservations": False, "vegetarian_options": False},
            {"name": "Odayasu", "cuisine": "Traditional Japanese",
             "price_range": "$$", "rating": 4.5,
             "specialty": "Tamagoyaki (egg), grilled fish, miso soup",
             "allergy_warning": "Fish present but no shellfish — check with staff",
             "hours": "6:00 AM - 3:00 PM", "wait_time": "30 min",
             "reservations": False, "vegetarian_options": True},
            {"name": "Turret Coffee Tsukiji", "cuisine": "Cafe/Light meals",
             "price_range": "$", "rating": 4.3,
             "specialty": "Coffee, pastries, sandwiches",
             "allergy_warning": "No seafood — safe for shellfish allergy",
             "hours": "7:00 AM - 6:00 PM", "wait_time": "None",
             "reservations": False, "vegetarian_options": True},
            {"name": "Tsukiji Kagura", "cuisine": "Japanese Kaiseki",
             "price_range": "$$$", "rating": 4.6,
             "specialty": "Multi-course kaiseki, seasonal ingredients",
             "allergy_warning": "Can accommodate shellfish allergy with 24hr notice",
             "hours": "12:00 PM - 10:00 PM", "wait_time": "Reservation recommended",
             "reservations": True, "vegetarian_options": True},
            {"name": "Nakamura Market Stall", "cuisine": "Street Food",
             "price_range": "$", "rating": 4.2,
             "specialty": "Tamagoyaki, dashi, grilled skewers (chicken/beef)",
             "allergy_warning": "Shellfish-free options available — ask vendor",
             "hours": "5:00 AM - 12:00 PM", "wait_time": "5-10 min",
             "reservations": False, "vegetarian_options": False}
        ],
        "shinjuku": [
            {"name": "Omoide Yokocho (Memory Lane)", "cuisine": "Yakitori",
             "price_range": "$$", "rating": 4.5,
             "specialty": "Grilled chicken skewers, beer, atmospheric alley dining",
             "allergy_warning": "No shellfish — safe for shellfish allergy",
             "hours": "5:00 PM - 12:00 AM", "vegetarian_options": False},
            {"name": "Ichiran Ramen Shinjuku", "cuisine": "Ramen",
             "price_range": "$", "rating": 4.6,
             "specialty": "Tonkotsu ramen, solo dining booths, customizable",
             "allergy_warning": "No shellfish in base broth — safe for shellfish allergy",
             "hours": "24 hours", "vegetarian_options": False},
            {"name": "New York Grill (Park Hyatt)", "cuisine": "International",
             "price_range": "$$$$", "rating": 4.7,
             "specialty": "Steaks, city views, Lost in Translation bar",
             "allergy_warning": "Full allergy accommodations available",
             "hours": "11:30 AM - 11:00 PM", "vegetarian_options": True}
        ],
        "kyoto": [
            {"name": "Nishiki Market stalls", "cuisine": "Japanese street food",
             "price_range": "$", "rating": 4.4,
             "specialty": "Pickles, tofu, matcha, local specialties",
             "allergy_warning": "Mixed — some seafood stalls, easily avoided",
             "hours": "9:00 AM - 6:00 PM", "vegetarian_options": True},
            {"name": "Kikunoi Honten", "cuisine": "Kaiseki",
             "price_range": "$$$$", "rating": 4.8,
             "specialty": "3 Michelin stars, traditional multi-course",
             "allergy_warning": "Accommodates allergies with advance notice",
             "hours": "12:00 PM - 10:00 PM", "vegetarian_options": True}
        ]
    }
    
    # Find restaurants for location
    location_lower = location.lower()
    restaurants = None
    for key in restaurant_database:
        if key in location_lower:
            restaurants = restaurant_database[key]
            break
    
    # Default restaurants if not in database
    if not restaurants:
        restaurants = [
            {"name": f"Local Restaurant 1 in {location}",
             "cuisine": "Local cuisine", "price_range": "$$",
             "rating": 4.2, "specialty": "Regional specialties",
             "allergy_warning": "Ask staff about specific allergens",
             "hours": "12:00 PM - 10:00 PM", "vegetarian_options": True},
            {"name": f"International Cafe in {location}",
             "cuisine": "International", "price_range": "$$",
             "rating": 4.0, "specialty": "Mixed international menu",
             "allergy_warning": "Allergy-aware kitchen, ask staff",
             "hours": "8:00 AM - 10:00 PM", "vegetarian_options": True}
        ]
    
    return {
        "search_type": "restaurants",
        "location": location,
        "results_count": len(restaurants),
        "results": restaurants,
        "allergy_note": (
            "⚠️ Always inform restaurant staff of any food allergies. "
            "The allergy_warning field provides general guidance only."
        ),
        "tips": [
            "Japanese restaurants often have picture menus — point to what you want",
            "Google Translate camera mode works well for Japanese menus",
            "Most restaurants appreciate knowing about allergies in advance"
        ]
    }


def _search_attractions(location: str) -> dict:
    """Generate realistic attraction search results."""
    location_lower = location.lower()
    
    attraction_database = {
        "tokyo": [
            {"name": "Senso-ji Temple", "area": "Asakusa", "type": "Temple/Cultural",
             "duration_hours": 1.5, "cost_usd": 0, "rating": 4.7,
             "best_time": "Early morning (6-8 AM) to avoid crowds",
             "accessibility": "Fully accessible"},
            {"name": "Shibuya Crossing", "area": "Shibuya", "type": "Landmark",
             "duration_hours": 0.5, "cost_usd": 0, "rating": 4.6,
             "best_time": "Evening rush hour (6-9 PM) for maximum effect",
             "accessibility": "Fully accessible"},
            {"name": "teamLab Borderless", "area": "Odaiba", "type": "Digital Art",
             "duration_hours": 3, "cost_usd": 32, "rating": 4.8,
             "best_time": "Weekday mornings", "accessibility": "Limited"},
            {"name": "Meiji Shrine", "area": "Harajuku", "type": "Shrine/Nature",
             "duration_hours": 1, "cost_usd": 0, "rating": 4.6,
             "best_time": "Morning", "accessibility": "Mostly accessible"},
            {"name": "Tokyo Skytree", "area": "Asakusa", "type": "Observatory",
             "duration_hours": 2, "cost_usd": 22, "rating": 4.5,
             "best_time": "Clear days, sunset", "accessibility": "Fully accessible"},
            {"name": "Tsukiji Fish Market (outer)", "area": "Tsukiji",
             "type": "Market/Food", "duration_hours": 2, "cost_usd": 0,
             "rating": 4.4, "best_time": "Early morning 6-8 AM",
             "accessibility": "Mostly accessible",
             "note": "Outer market only — inner auction requires early booking"}
        ],
        "kyoto": [
            {"name": "Fushimi Inari Shrine", "area": "Fushimi", "type": "Shrine",
             "duration_hours": 2, "cost_usd": 0, "rating": 4.8,
             "best_time": "Early morning or evening", "accessibility": "Steps involved"},
            {"name": "Arashiyama Bamboo Grove", "area": "Arashiyama", "type": "Nature",
             "duration_hours": 1.5, "cost_usd": 0, "rating": 4.6,
             "best_time": "Early morning (7-8 AM)", "accessibility": "Flat paths"},
            {"name": "Kinkaku-ji (Golden Pavilion)", "area": "Northwest Kyoto",
             "type": "Temple", "duration_hours": 1, "cost_usd": 5, "rating": 4.7,
             "best_time": "Morning when opens", "accessibility": "Mostly accessible"},
            {"name": "Gion District Walk", "area": "Gion", "type": "Cultural/Walking",
             "duration_hours": 2, "cost_usd": 0, "rating": 4.5,
             "best_time": "Dusk (geisha most likely seen 6-8 PM)",
             "accessibility": "Flat cobblestone"}
        ],
        "paris": [
            {"name": "Eiffel Tower", "area": "7th arr", "type": "Landmark",
             "duration_hours": 2, "cost_usd": 28, "rating": 4.7,
             "best_time": "Sunset or night for light show",
             "accessibility": "Elevator available"},
            {"name": "Louvre Museum", "area": "1st arr", "type": "Museum",
             "duration_hours": 4, "cost_usd": 22, "rating": 4.7,
             "best_time": "Wednesday/Friday evenings (open late)",
             "accessibility": "Fully accessible"},
            {"name": "Montmartre & Sacré-Cœur", "area": "18th arr", "type": "Neighborhood",
             "duration_hours": 3, "cost_usd": 0, "rating": 4.6,
             "best_time": "Morning for views", "accessibility": "Funicular available"},
            {"name": "Seine River Cruise", "area": "City Center", "type": "Tour",
             "duration_hours": 1.5, "cost_usd": 18, "rating": 4.5,
             "best_time": "Sunset", "accessibility": "Fully accessible"}
        ]
    }
    
    # Find attractions for location
    attractions = None
    for key in attraction_database:
        if key in location_lower:
            attractions = attraction_database[key]
            break
    
    if not attractions:
        attractions = [
            {"name": f"City Center of {location}", "type": "Sightseeing",
             "duration_hours": 2, "cost_usd": 0, "rating": 4.0,
             "best_time": "Morning", "accessibility": "Varies"}
        ]
    
    total_cost = sum(a["cost_usd"] for a in attractions)
    total_time = sum(a["duration_hours"] for a in attractions)
    
    return {
        "search_type": "attractions",
        "location": location,
        "results_count": len(attractions),
        "results": attractions,
        "planning_summary": {
            "total_attractions": len(attractions),
            "total_estimated_hours": total_time,
            "total_estimated_cost_usd": total_cost,
            "days_needed_at_2_per_day": len(attractions) / 2,
            "days_needed_at_3_per_day": len(attractions) / 3
        },
        "tips": [
            "Book popular attractions online in advance",
            "Many temples/shrines are free or very cheap",
            "Early morning visits avoid crowds and heat"
        ]
    }