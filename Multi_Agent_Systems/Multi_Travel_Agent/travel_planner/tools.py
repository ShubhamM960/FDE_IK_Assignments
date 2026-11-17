"""Tools available to the agents.

    flight_search                  -> SerpAPI Google Flights
    hotel_search                   -> SerpAPI Google Hotels
    tavily_search                  -> Tavily web search
    call_itinerary_research_agent  -> wraps a Deep Agent that builds itineraries

Tool groups at the bottom are what each agent is allowed to use.
"""

import json
import os
from typing import Optional

from deepagents import create_deep_agent
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

from .config import SERPAPI_API_KEY, llm
from .prompts import ITINERARY_RESEARCH_PROMPT


def _serpapi_search(params: dict) -> dict:
    """Call SerpAPI. Imported lazily so the app still starts if the package is missing."""
    from serpapi import GoogleSearch  # pip install google-search-results

    return GoogleSearch({**params, "api_key": SERPAPI_API_KEY, "hl": "en", "currency": "USD"}).get_dict()


# ===========================================================================
# 1. Flight search
# ===========================================================================
class FlightSearchInput(BaseModel):
    """Input schema for flight search."""

    departure_airport: str = Field(..., description="Departure airport IATA code (e.g., 'DEL')")
    arrival_airport: str = Field(..., description="Arrival airport IATA code (e.g., 'LHR')")
    outbound_date: str = Field(..., description="Departure date (YYYY-MM-DD)")
    return_date: Optional[str] = Field(None, description="Return date (YYYY-MM-DD) for round trips")
    adults: int = Field(1, description="Number of adults")
    children: int = Field(0, description="Number of children")
    stops: Optional[int] = Field(None, description="0=Any, 1=Nonstop, 2=1 stop or fewer, 3=2 stops or fewer")


@tool(args_schema=FlightSearchInput)
def flight_search(
    departure_airport: str,
    arrival_airport: str,
    outbound_date: str,
    return_date: Optional[str] = None,
    adults: int = 1,
    children: int = 0,
    stops: Optional[int] = None,
) -> str:
    """Search Google Flights for flights between two airports on a given date."""
    params = {
        "engine": "google_flights",
        "departure_id": departure_airport.upper(),
        "arrival_id": arrival_airport.upper(),
        "outbound_date": outbound_date,
        "adults": adults,
        "children": children,
        "type": "1" if return_date else "2",  # 1 = round trip, 2 = one way
    }
    if return_date:
        params["return_date"] = return_date
    if stops is not None:
        params["stops"] = stops

    try:
        results = _serpapi_search(params)
        if "error" in results:
            return json.dumps({"message": results["error"]})

        options = results.get("best_flights", []) + results.get("other_flights", [])
        if not options:
            return json.dumps({"message": "No flights found"})

        output = []
        for option in options[:10]:
            segments = option.get("flights", [])
            output.append({
                "price_usd": option.get("price"),
                "total_duration_min": option.get("total_duration"),
                "stops": max(len(segments) - 1, 0),
                "layovers": [l.get("name") for l in option.get("layovers", [])],
                "segments": [
                    {
                        "airline": s.get("airline"),
                        "flight_number": s.get("flight_number"),
                        "from": s.get("departure_airport", {}).get("id"),
                        "departure_time": s.get("departure_airport", {}).get("time"),
                        "to": s.get("arrival_airport", {}).get("id"),
                        "arrival_time": s.get("arrival_airport", {}).get("time"),
                        "travel_class": s.get("travel_class"),
                    }
                    for s in segments
                ],
            })
        return json.dumps(output)
    except Exception as e:
        return json.dumps({"message": f"Error searching for flights: {e}"})


# ===========================================================================
# 2. Hotel search
# ===========================================================================
class HotelSearchInput(BaseModel):
    """Input schema for hotel search."""

    location: str = Field(..., description="City or area to search hotels in (e.g., 'London')")
    check_in_date: str = Field(..., description="Check-in date (YYYY-MM-DD)")
    check_out_date: str = Field(..., description="Check-out date (YYYY-MM-DD)")
    adults: int = Field(1, description="Number of adults")
    children: int = Field(0, description="Number of children")
    rooms: int = Field(1, description="Number of rooms (informational)")


@tool(args_schema=HotelSearchInput)
def hotel_search(
    location: str,
    check_in_date: str,
    check_out_date: str,
    adults: int = 1,
    children: int = 0,
    rooms: int = 1,
) -> str:
    """Search Google Hotels for hotels in a location for the given check-in / check-out dates."""
    params = {
        "engine": "google_hotels",
        "q": f"hotels in {location}",
        "check_in_date": check_in_date,
        "check_out_date": check_out_date,
        "adults": adults,
        "children": children,
    }
    try:
        results = _serpapi_search(params)
        if "error" in results:
            return json.dumps({"message": results["error"]})

        hotels = results.get("properties", [])
        if not hotels:
            return json.dumps({"message": "No hotels found"})

        output = [
            {
                "name": h.get("name"),
                "hotel_class": h.get("hotel_class"),
                "rating": h.get("overall_rating"),
                "reviews": h.get("reviews"),
                "price_per_night": h.get("rate_per_night", {}).get("lowest"),
                "total_price": h.get("total_rate", {}).get("lowest"),
                "amenities": h.get("amenities", [])[:8],
                "link": h.get("link"),
            }
            for h in hotels[:10]
        ]
        return json.dumps(output)
    except Exception as e:
        return json.dumps({"message": f"Error searching for hotels: {e}"})


# ===========================================================================
# 3. Web search (Tavily)
# ===========================================================================
tavily_search = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced",
    # A dummy value keeps imports working when the key is missing; the UI warns instead.
    tavily_api_key=os.getenv("TAVILY_API_KEY") or "not-set",
)


# ===========================================================================
# 4. Itinerary research Deep Agent, exposed as a tool
# ===========================================================================
itinerary_research_deep_agent = create_deep_agent(
    model=llm,
    system_prompt=ITINERARY_RESEARCH_PROMPT,
    tools=[tavily_search],
)


@tool
def call_itinerary_research_agent(query: str) -> str:
    """Create a detailed day-by-day travel itinerary. Pass the full trip request
    (destination, number of days, traveller type, interests, budget)."""
    result = itinerary_research_deep_agent.invoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    return result["messages"][-1].content


# ===========================================================================
# Tool groups (what each agent is allowed to use)
# ===========================================================================
hotel_flight_tools = [flight_search, hotel_search]
travel_scout_tools = [tavily_search, call_itinerary_research_agent]
