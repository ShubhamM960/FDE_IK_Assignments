# Necessary Imports
import operator
import os
from typing import Literal

from langchain.chat_models import init_chat_model

from dotenv import load_dotenv

load_dotenv()



# LLM
from langchain_core.language_models import llms
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    # stream_usage=True,
    temperature=0.2,
    # max_tokens=None,
    # timeout=None,
    # reasoning_effort="low",
    # max_retries=2,
    # api_key="...",  # If you prefer to pass api key in directly
    # base_url="...",
    # organization="...",
    # other params...
)
##############
#####PART - 1: 
##############
                ########################################################                                
                    ####***** Flight & Hotel Search Agent ******
                ########################################################    
#necessary imports for part - 1
from serpapi import GoogleSearch
import os
import json
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain.agents import create_agent

# FLIGHT SEARCH TOOL
class FlightSearchInput(BaseModel):
    """Input schema for flight search."""

    departure_airport: str = Field(..., description="Departure airport code (e.g., 'JFK')")
    arrival_airport: str = Field(..., description="Arrival airport code (e.g., 'LAX')")
    outbound_date: str = Field(..., description="Departure date (YYYY-MM-DD)")
    return_date: Optional[str] = Field(None, description="Return date (YYYY-MM-DD)")
    adults: int = Field(1, description="Number of adults")
    children: int = Field(0, description="Number of children")
    stops: Optional[int] = Field(None, description="0=Any, 1=Nonstop, 2=1 stop or fewer")

# @tool
# def flight_search(input: FlightSearchInput) -> str:
#     """Search for flights between two airports on a given date."""
#     return f"Searching for flights from {input.departure_airport} to {input.arrival_airport} on {input.outbound_date}."

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
    """Search for flights between two airports on a given date."""
    params = {
        "engine": "google_flights",
        "api_key": os.getenv("SERPAPI_API_KEY"),
        "departure_id": departure_airport,
        "arrival_id": arrival_airport,
        "outbound_date": outbound_date,
        "type": "1" if return_date else "2",  # 1 = round trip, 2 = one way
        "adults": adults,
        "children": children,
        "currency": "USD",
        "hl": "en",
    }
    if return_date:
        params["return_date"] = return_date
    if stops is not None:
        params["stops"] = stops

    try:
        search = GoogleSearch(params)
        results = search.get_dict()

        if "error" in results:
            return json.dumps({"message": f"SerpApi error: {results['error']}"})

        flights_results = results.get("best_flights", []) + results.get("other_flights", [])
        if not flights_results:
            return json.dumps({"message": "No flights found"})

        output = []
        for option in flights_results[:10]:
            legs = option.get("flights", [])
            output.append({
                "price": option.get("price"),
                "total_duration_minutes": option.get("total_duration"),
                "stops": max(len(legs) - 1, 0),
                "legs": [
                    {
                        "airline": leg.get("airline"),
                        "flight_number": leg.get("flight_number"),
                        "from": leg.get("departure_airport", {}).get("id"),
                        "departure_time": leg.get("departure_airport", {}).get("time"),
                        "to": leg.get("arrival_airport", {}).get("id"),
                        "arrival_time": leg.get("arrival_airport", {}).get("time"),
                        "duration_minutes": leg.get("duration"),
                    }
                    for leg in legs
                ],
            })

        return json.dumps(output)
    except Exception as e:
        return json.dumps({"message": f"Error searching for flights: {e}"})



# HOTEL SEARCH TOOL
class HotelSearchInput(BaseModel):
    """Input schema for hotel search."""
    location: str = Field(..., description="Location of the hotel (e.g., 'New York, NY')")
    check_in_date: str = Field(..., description="Check-in date (YYYY-MM-DD)")
    check_out_date: str = Field(..., description="Check-out date (YYYY-MM-DD)")
    adults: int = Field(1, description="Number of adults")
    children: int = Field(0, description="Number of children")
    rooms: int = Field(1, description="Number of rooms")

@tool(args_schema=HotelSearchInput)
def hotel_search(
    location: str,
    check_in_date: str,
    check_out_date: str,
    adults: int = 1,
    children: int = 0,
    rooms: int = 1,
) -> str:
    """Search for hotels in a given location on a given date."""
    # Note: Google Hotels has no "rooms" parameter, and "children" also needs
    # "children_ages", so only adults are sent to the API.
    params = {
        "engine": "google_hotels",
        "api_key": os.getenv("SERPAPI_API_KEY"),
        "q": f"hotels in {location}",
        "check_in_date": check_in_date,
        "check_out_date": check_out_date,
        "adults": adults,
        "currency": "USD",
        "hl": "en",
    }
    try:
        search = GoogleSearch(params)
        results = search.get_dict()

        if "error" in results:
            return json.dumps({"message": f"SerpApi error: {results['error']}"})

        hotels_results = results.get("properties", [])
        if not hotels_results:
            return json.dumps({"message": "No hotels found"})
        output = []
        for hotel in hotels_results[:10]:
            output.append({
                "name": hotel.get("name"),
                "price_per_night": hotel.get("rate_per_night", {}).get("lowest"),
                "total_price": hotel.get("total_rate", {}).get("lowest"),
                "rating": hotel.get("overall_rating"),
                "hotel_class": hotel.get("hotel_class"),
                "link": hotel.get("link"),
            })
        return json.dumps(output)
    except Exception as e:
        return json.dumps({"message": f"Error searching for hotels: {e}"})

# Create a list of tools
hotel_flight_search_tools = [flight_search, hotel_search]

#flight hotel search agent instructions
hotel_flight_search_instructions = """You are a travel search agent. You are given a user's travel request and you need to search for the best flights and hotels for the user.
You only handle hotel and flight search or combined hotel and flight search requests.

SCOPE AND BEHAVIOR RULES
- Respond ONLY to travel-related requests, for hotel and flighr search.
- Use the appropriate tools provided to search for the best flights and hotels for the user.
- If a request is unrelated to travel (e.g., math, general knowledge, coding, weather outside trip context), politely decline and redirect .
- Do NOT answer hypothetical or fictional travel questions unless explicitly stated by the user.

CLARIFICATION GATE 
- If any information is missing, ask the user for the missing information and continue to ask untill you confirm  with the requirements
- do not partially execute the request with assumed data it may provide incorrect results.

TOOL USAGE
- FlightSearch is the tool for searching for flights.
- HotelSearch is the tool for searching for hotels.
- You can use both tools together to search for flights and hotels.

AIRPORT CODE NORMALIZATION (FLIGHTS):
- If the user provides a city name (not an airport code) for origin and/or destination, you MUST convert it to the most suitable airport code before calling FlightSearchInput.
- Prefer the primary international airport for that city (or the best-served airport for typical commercial routes).
- If the city has multiple major airports (e.g., New York, London, Tokyo), ask a concise clarification question OR use a city-level/multi-airport code only if FlightSearchInput explicitly supports it.
- If you cannot confidently determine the correct airport code from the city name, ask a clarification question and STOP (do not guess).

Airport code reference (examples):
- Delhi: DEL
- London Heathrow: LHR
- New York: JFK / LGA / EWR
- etc.


OUTPUT REQUIREMENTS

For Flight Results:
- Organize by outbound and return (if applicable)
- Include airline, timing, duration, layovers, cabin class, and price
- Highlight best options (e.g., fastest, cheapest, best value)

For Hotel Results:
- Include available hotels with their name, star rating (if available), guest rating in the city
- Price per night and estimated total
- Key amenities and location details
- Booking recommendations

For Combined Requests:
- Provide bundled recommendations (e.g., Value / Balanced / Comfort)
- Show estimated total trip cost (flight + hotel)
- Explain trade-offs clearly

FORMATTING GUIDELINES:
- Use clear headings and bullet points
- Optimize for quick comparison and decision-making
- Be concise, factual, and analytical
"""

#create hotel flight search agent
hotel_flight_search_agent = create_agent(
    model=llm,
    system_prompt=hotel_flight_search_instructions,
    tools=hotel_flight_search_tools,
)

# ************ TESTING HOTEL FLIGHT SEARCH AGENT ************
# hotel_flight_search_response = hotel_flight_search_agent.invoke({"input": "Flights from Delhi to London on 2026-12-10 for 2 adults"})
# hotel_flight_search_response = hotel_flight_search_agent.invoke({"input": "Hotels in London on 2026-12-10 for 2 adults"})
# hotel_flight_search_response = hotel_flight_search_agent.invoke({"input": "Flights from Delhi to London on 2026-12-10 for 2 adults and Hotels in London on 2026-12-10 for 2 adults"})

# ************ *************************** ************

##############
#####PART - 2: 
##############
        ##################################
        ####****** WEB SEARCH TOOL ******
        ##################################
#necessary imports for part - 2
from tavily import TavilyClient
from langchain_tavily import TavilySearch
from deepagents import create_deep_agent


# Initialize the Tavily client
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
print('--' * 50)


# Create Internet Search vis TavilySearch
tavily_search = TavilySearch(
    max_results=5,
    topic="general",
    # include_answer=False,
    # include_raw_content=False,
    include_images=True,
    include_image_descriptions=True,
    search_depth="advanced",
    # time_range="day",
    #include_domains=None,
    # exclude_domains=None
)

        ########################################################
        ####****** Create Deep Agent ******
        ########################################################

research_deep_agent_instructions = """You are a professional travel itinerary planning agent specializing exclusively in trip research and itinerary design.

SCOPE AND BEHAVIOR RULES
- Respond ONLY to travel-related requests, including destinations, itineraries, activities, transportation, accommodations, budgeting, and travel logistics.
- If a request is unrelated to travel (e.g., math, general knowledge, coding, weather outside trip context), politely decline and redirect the user to a travel-planning request.
- Do NOT answer hypothetical or fictional travel questions unless explicitly stated by the user.

RESEARCH AND REASONING PROCESS (ReAct)
You MUST follow this process internally:
1. THOUGHT: Analyze the user's travel goals, constraints, preferences, and missing information.
2. ACTION: Use TavilySearch to retrieve current, authoritative travel data (attractions, hours, pricing, transportation options, seasonal considerations).
3. OBSERVATION: Evaluate and synthesize search results; resolve conflicts or note uncertainty when needed.
4. RESPONSE: Produce a complete, user-ready itinerary.

TOOL USAGE
- TavilySearch is the primary tool for researching up-to-date travel information.
- Prefer official tourism boards, transportation providers, reputable travel guides, and recent reviews.
- Do not fabricate details if information is unavailable; explicitly state assumptions or gaps.

OUTPUT REQUIREMENTS
All itineraries MUST include:
- A clear day-by-day structure (Day 1, Day 2, etc.)
- Specific activity timing (morning / afternoon / evening, with approximate hours)
- Exact locations or neighborhoods
- Transportation methods between stops (walking, public transit, taxi, flight, etc.)
- Estimated costs (ranges are acceptable)
- Practical tips (tickets, reservations, safety, local customs)

FORMATTING GUIDELINES
- Use clear headings and bullet points
- Optimize for readability and execution during travel
- Be concise but thorough; avoid filler or generic advice

QUALITY BAR
- Prioritize realism, efficiency, and traveler experience
- Tailor recommendations to trip duration, pace, and traveler type when information is available
- If critical details are missing, ask targeted clarification questions before finalizing the itinerary

"""

#model for Deep Agent
model = init_chat_model(model="gpt-4o-mini", model_provider="openai", temperature=0.2)
# Itinerary Agent
itinerary_research_deep_agent = create_deep_agent(
    model=model,
    system_prompt=research_deep_agent_instructions,
    tools=[tavily_search],
)

    ######################################
    #***** Create travel scout agent ******
    ######################################

#call itinerary research agent
@tool
def call_itinerary_research_agent(query: str) -> str:
    """Call the itinerary research agent to research the query."""
    return itinerary_research_deep_agent.invoke({"input": query}) 


travel_scout_instructions = """You are a travel scout agent. 
role : you are  are given a user's travel request and travel planning and you need to research details provide highlevel travel information and guidance.

SCOPE AND RESPONSIBILITIES
- Respond only to travel-related requests, such as :
  - Weather and climate of destinations
  - Best cities or regions to visit by season or interest
  - What to pack or wear (clothing, gear, cultural norms)
  - Safety, visas, currency, local customs, and basic logistics
  - High-level comparisons between destinations
- To create day-by-day itineraries if asked use **itinerary_research_deep_agent** .
- Do NOT search or recommend specific flights or hotels.
- Politely decline and redirect if the request is unrelated to travel.

--------------------------------------------------
TOOL SELECTION RULES (CRITICAL)
--------------------------------------------------
- Identify the intent and depth of the request of travel question and use the appropriate tool.


USE **tavily_search** for:
- General travel questions
- High-level guidance and quick factual lookups
- Topics that do NOT require structured planning or multi-day sequencing

Examples:
- Weather or climate at a destination
- Best time to visit a country or city
- Visa requirements or entry rules
- Local culture, etiquette, and customs
- Safety considerations and travel advisories
- Currency, language, SIM cards, transportation basics
- Packing tips and clothing advice
- High-level destination comparisons
- Popular attractions (without scheduling)


USE **itinerary_research_deep_agent** for:
- Any request that requires structured planning or multi-day sequencing
- Creating day-by-day or multi-day travel plans
- Scheduling activities and attractions
- Detailed travel logistics (transportation, accommodations
- Detailed cost estimates and budgeting
- Practical travel tips and recommendations
If the user explicitly asks for:
- A daily schedule
- A detailed itinerary
- A multi-city route plan
→ You MUST use itinerary_research_agent.

EXAMPLES:
- Create a day-by-day itinerary for a trip to Tokyo
- Plan a 7 day trip to Paris with specific activities and attractions
- Create a budget-friendly 3 day itinerary for a trip to London
- Create a 7 days honymoon trip to Italy
- “What should I do each day in Bali for 5 days?”

Clarification Gate:
- If any information is missing or not clear about the requirements ask the user for the information and continue to ask untill you confirm  with the requirements or you have enough information to proceed.
- do not partially execute the request with assumed data it may provide incorrect results.


OUTPUT REQUIREMENTS:
- Provide high-level travel information and guidance.
- Provide a clear and concise answer to the user's travel question.
- Provide a detailed itinerary if requested.
- Provide a clear and concise answer to the user's travel question.

SOURCE CITATION:
- Always cite your sources when providing factual information at the end in 'Sources' section.
- Include the relevant 2-3 sources for each information provided via TavilySearch.
- Do not include raw URLs in the body.
- List out the resource clearly one by one.
like this :
*** Sources *** 
- [Title] (URL1)
- [Title] (URL2)
- [Title] (URL3)
This helps users verify information and explore further.

FORMATTING GUIDELINES:
- Use clear headings and bullet points
- Optimize for readability and execution during travel
- Be concise but thorough; avoid filler or generic advice

QUALITY BAR:
- Prioritize realism, efficiency, and traveler experience
- Tailor recommendations to trip duration, pace, and traveler type when information is available
- Optimize the general guidance not trip execution details.
- Explicitly state assumptions or seasonal variations when applicable

"""

#create travel scout agent
travel_scout_agent = create_agent(
    model=llm,
    system_prompt=travel_scout_instructions,
    tools=[call_itinerary_research_agent, tavily_search]
)

# ************ TESTING TRAVEL SCOUT AGENT ************

# travel_scout_agent_response1 = travel_scout_agent.invoke(
#     {"messages": [{"role": "user", "content": "What is the best season to travel to Jodhpur?"}]})

# travel_scout_agent_response2 = travel_scout_agent.invoke(
#     {"input": "Plan 2 day itinerary for Agra travel for a history & culture lover, also want to try special cusine to this city"})

# for event in travel_scout.stream(
#     {"messages": [{"role": "user", "content": "What is the best season to travel to Jodhpur?"}]},
#     stream_mode="values",):
#       print(event)
#       event["messages"][-1].pretty_print()
# ************ *************************** ************

########################################################################################
###########************* INVOKE  LANGCHAIN SUB-AGENTS *************###########
########################################################################################
#hotel_flight_search_agent and travel_scout_agent these will be act as sub-agents 
#to dynamically call and cordinate with these two agents we will use main_agent which will be the router agent.
# @tool
# def call_travel_flight_agent(query: str) -> str:
#     return hotel_flight_search_agent.invoke({"input": query})
# @tool
# def call_travel_scout_agent(query: str) -> str:
#     return travel_scout_agent.invoke({"input": query})

main_agent_instructions = """ You are a router agent for a travel assistant system.
You do NOT answer travel questions yourself. Your only job is to analyse the user's request,
split it into tasks, and route each task to the correct agent.


SCOPE AND RESPONSIBILITIES
- Respond to travel related request or search for flights and hotels.
- If the request is not travel related or hotel and flight search request politely decline request.
- Do NOT answer hypothetical or fictional travel questions .

TOOL USAGE
- call_travel_flight_agent : is the tool for searching for flights and hotels.
- call_travel_scout_agent : is the tool for searching for travel guidance / requires structured planning or multi-day sequencing.

TASK DECOMPOSITION
- For each task, rewrite user_query so it contains ONLY the part relevant to that agent,
  including all details the user provided (dates, travellers, preferences, budget).
- Set focus to a short phrase describing what the agent should prioritise
  (e.g. "cheapest nonstop flights", "4-star hotels near city centre", "history & culture itinerary").

EXAMPLES
- "Best time to visit Jodhpur?" -> call_travel_scout_agent
- "Flights from Delhi to London on 2026-12-10 for 2 adults" -> call_travel_flight_agent
- "Plan 2 days in Agra and find a hotel there for 12-13 Dec" -> call_travel_scout_agent

CLARIFICATION GATE
- Before routing, make sure the required information is present:
  - Flights: origin, destination, outbound date (return date, adults, children, stops if relevant)
  - Hotels: location, check-in date, check-out date (adults, children, rooms if relevant)
  - Itinerary: destination, trip duration (traveller type/interests if available)
- If anything required is missing or unclear, ask the user and continue asking until confirmed.
- Do not partially execute the request with assumed data; it may produce incorrect results.

DATA NORMALIZATION
- Convert dates to YYYY-MM-DD format.
- City names are acceptable; do not guess airport codes (the search agent handles them).


OUTPUT REQUIREMENTS
- Provide a clear and concise answer to the user's travel question.
- Provide a detailed itinerary if requested.

"""
############################################################################################
################*************** Langchain main agent creation ***************################
############################################################################################
# main_agent = create_agent(
#     model=llm,
#     system_prompt=main_agent_instructions,
#     tools=[call_travel_flight_agent, call_travel_scout_agent]
# )

# ************ TESTING MAIN AGENT ************
# main_agent_response = main_agent.invoke({"messages": [{"role": "user", "content": "Best time to visit Jodhpur?"}]})

# main_agent_response = main_agent.invoke({"messages": [{"role": "user", "content": "Flights from Delhi to London on 2026-12-10 for 2 adults"}]})
# main_agent_response = main_agent.invoke({"messages": [{"role": "user", "content": "Plan 2 days in Agra and find a hotel there for 12-13 Dec"}]})

# ************ *************************** ************







########################################################################################
###########************* DESIGN USING LANGGRAPH , STATES and  NODES *************###########
########################################################################################
# Augment the LLM with tools
hotel_flight_search_tools_by_name = {tool.name: tool for tool in hotel_flight_search_tools}
hotel_flight_search_model_with_tools = llm.bind_tools(hotel_flight_search_tools) #hotel_flight_search_agent's tools

travel_scout_tools = [call_itinerary_research_agent, tavily_search]
travel_scout_tools_by_name = {tool.name: tool for tool in travel_scout_tools}
travel_scout_model_with_tools = llm.bind_tools(travel_scout_tools) #travel_scout_agent's tools


# Graph initial state 
from langgraph.graph import MessagesState
from langgraph.types import Send
from typing import Annotated, List, Literal, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage, AIMessage
import operator


class AgentTask(BaseModel):
    """A single agent task with targeted query."""
    source: Literal["flight_hotel_search_agent_node", "travel_scout_agent_node"]
    user_query: str
    focus: str = ""  # What aspect this agent should focus on

class TravelPlannerState(MessagesState):
    """Enhanced state for parallel execution."""

    # Current user query
    user_query: str

    # Tasks from router (replaces single classification)
    tasks: List[AgentTask]

    # Flag for synthesis for single agent or multiple agents
    requires_synthesis: bool

    # Parallel agent results - KEY: uses operator.add for concurrent writes
    agent_results: Annotated[List[dict], operator.add]

    # Final synthesized answer
    final_answer: str

    search_messages: Annotated[List[AnyMessage], operator.add]
    planner_messages: Annotated[List[AnyMessage], operator.add]

class ClassificationResult(BaseModel):
    """Router output - now supports MULTIPLE parallel tasks."""
    tasks: List[AgentTask] = Field(
        description="List of agents to invoke with their targeted queries"
    )
    requires_synthesis: bool = Field(
        default=False,
        description="Whether multiple agents are being used and synthesis is needed"
    )

# Worker/Node state for individual agents
class AgentWorkerState(TypedDict):
    """State passed to each parallel worker."""
    task: AgentTask
    messages: List[AnyMessage]
    agent_results: Annotated[List[dict], operator.add]
###################################
#hotel flight search nodes building
###################################

def hotel_flight_agent_node(state : TravelPlannerState) -> str:
    search_messages = state.get('search_messages', [])

    messages = [SystemMessage(content=hotel_flight_search_instructions)] + search_messages

    #if it is first call, add the user query to the messages (and save it to state)
    new_messages = []
    if not search_messages :
        user_msg = HumanMessage(content=state.get('user_query', ''))
        messages.append(user_msg)
        new_messages.append(user_msg)

    result = hotel_flight_search_model_with_tools.invoke(messages)
    
    return {
        'search_messages' : new_messages + [result],
        'agent_results' : [{
            'agent_name' : 'hotel_flight_search_agent',
            'result' : result.content
        }] if not result.tool_calls else []
        
    }
  
def hotel_flight_tool_node(state : TravelPlannerState) -> str:
    search_messages = state.get('search_messages', [])

    if not search_messages :
        return {'search_messages' : []}

    if not search_messages[-1].tool_calls:
        return {'search_messages' : []}
    
    tool_results = []
    for tool_call in search_messages[-1].tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        tool_function = hotel_flight_search_tools_by_name.get(tool_name)
        tool_output = tool_function.invoke(tool_args)
        print(f"tool output: {tool_output}")
        tool_results.append(
            ToolMessage(content=str(tool_output), tool_call_id=tool_call['id'])
        )

    return {
        'search_messages' : tool_results
    }


def should_continue_for_hotel_flight(state : TravelPlannerState) -> bool:
    search_messages = state.get('search_messages', [])
    if search_messages and search_messages[-1].tool_calls:
        return 'hotel_flight_tool_node'
    else :
        return 'synthesizer'

###################################
#travel scout nodes building
###################################
def travel_scout_agent_node(state : TravelPlannerState) -> str:
    planner_messages = state.get('planner_messages', [])

    messages = [SystemMessage(content=travel_scout_instructions)] + planner_messages
    #if it is first call, add the user query to the messages (and save it to state)
    new_messages = []
    if not planner_messages :
        user_msg = HumanMessage(content=state.get('user_query', ''))
        messages.append(user_msg)
        new_messages.append(user_msg)

    result = travel_scout_model_with_tools.invoke(messages)
    
    return {
        'planner_messages' : new_messages + [result],
        'agent_results' : [{
            'agent_name' : 'travel_scout_agent',
            'result' : result.content
        }] if not result.tool_calls else []
        
    }

def travel_scout_tool_node(state : TravelPlannerState) -> str:
    planner_messages = state.get('planner_messages', [])
    if not planner_messages :
        return {'planner_messages' : []}
    if not planner_messages[-1].tool_calls:
        return {'planner_messages' : []}
    tool_results = []
    for tool_call in planner_messages[-1].tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        tool_function = travel_scout_tools_by_name.get(tool_name)
        tool_output = tool_function.invoke(tool_args)
        print(f"tool output information is  \n\n tool name: {tool_name} \n tool output: {tool_output}")
        tool_results.append(
            ToolMessage(content=str(tool_output), tool_call_id=tool_call['id'])
        )
    return {
        'planner_messages' : tool_results
    }

def should_continue_for_travel_scout(state : TravelPlannerState) -> str:
    planner_messages = state.get('planner_messages', [])
    if planner_messages and planner_messages[-1].tool_calls:
        return 'travel_scout_tool_node'
    else :
        return 'synthesizer'

def orchestrator(state : TravelPlannerState) -> str:
    """
    Updated router that can dispatch to MULTIPLE agents.
    """
    router_llm = llm
    structured_llm = router_llm.with_structured_output(ClassificationResult)

    result = structured_llm.invoke([
        {
            "role": "system",
            "content": """ You are a Travel Request Router that can dispatch to MULTIPLE agents in parallel.

    AVAILABLE AGENTS:
    1. flight_hotel_search_agent_node - Flight searches, hotel searches, pricing, availability
    2. travel_scout_agent_node - Itineraries, travel tips, weather, visas, activities, general guidance

    ROUTING RULES:

    SINGLE AGENT (requires_synthesis = false):
    - Flight/hotel ONLY → flight_hotel_search_agent_node
    - Itinerary/planning ONLY → travel_scout_agent_node
    - General travel questions → travel_scout_agent_node

    MULTIPLE AGENTS (requires_synthesis = true):
    - "Plan a trip AND find flights/hotels" → BOTH agents
    - "Find hotels and create itinerary" → BOTH agents
    - Requests mentioning BOTH booking AND planning → BOTH agents

    When dispatching to BOTH agents:
    1. Create a focused sub-query for each agent
    2. flight_hotel_search_agent_node query should focus on flights/hotels/pricing
    3. travel_scout_agent_node query should focus on itinerary/activities/tips

    EXAMPLES:

    Query: "Find flights from DEL to DXB"
    → tasks: [{"source": "flight_hotel_search_agent_node", "user_query": "Find flights from DEL to DXB"}]
    → requires_synthesis: false

    Query: "Plan a 5-day trip to Paris"
    → tasks: [{"source": "travel_scout_agent_node", "user_query": "Plan a 5-day trip to Paris"}]
    → requires_synthesis: false

    Query: "Plan a week in London and find me hotels under $200/night"
    → tasks: [
        {"source": "flight_hotel_search_agent_node", "user_query": "Find hotels in London under $200/night for 1 week", "focus": "hotel search and pricing"},
        {"source": "travel_scout_agent_node", "user_query": "Plan a week-long itinerary for London", "focus": "daily activities and attractions"}
    ]
    → requires_synthesis: true

    Query: "I want to visit Tokyo - book flights from NYC and create a 5-day itinerary"
    → tasks: [
        {"source": "flight_hotel_search_agent_node", "user_query": "Find flights from NYC to Tokyo", "focus": "flight options and prices"},
        {"source": "travel_scout_agent_node", "user_query": "Create a 5-day itinerary for Tokyo", "focus": "daily schedule and activities"}
    ]
    → requires_synthesis: true """
            },
            {"role": "user", "content": state["user_query"]}
        ])

    return {
            "tasks": result.tasks,
            "requires_synthesis": result.requires_synthesis
        }

def dispatch_to_agents(state : TravelPlannerState) :
    tasks = state['tasks']

    send_calls = []
    for task in tasks:
        agent_state={
            'user_query' : task.user_query,
            'focus' : task.focus,
            'source' : task.source,
            'requires_synthesis' : state.get('requires_synthesis', False),
            'messages' : state.get('messages', []),
            'agent_results' : state.get('agent_results', []),
            'search_messages' : state.get('search_messages', []),
            'planner_messages' : state.get('planner_messages', [])
        }
        if task.source == 'flight_hotel_search_agent_node':
            send_calls.append(Send('hotel_flight_agent_node', agent_state))
        elif task.source == 'travel_scout_agent_node':
            send_calls.append(Send(travel_scout_agent_node, agent_state))

    return send_calls


def synthesizer_node(state: TravelPlannerState) -> dict:
    """
    Combines results from parallel agents into a unified response.
    Only called when requires_synthesis is True.
    """

    agent_results = state.get("agent_results", [])

    if not agent_results:
        return {
            "final_answer": "I couldn't process your request.",
            "messages": [AIMessage(content="I couldn't process your request.")]
        }

    # If only one result, no synthesis needed
    if len(agent_results) == 1:
        result = agent_results[0]["result"]
        return {
            "final_answer": result,
            "messages": [AIMessage(content=result)]
        }

    # Format results for synthesis
    results_formatted = "\n\n" + "="*50 + "\n\n".join([
        f"**{r['agent_name'].upper()}**\nFocus: {r.get('focus', 'N/A')}\n\n{r['result']}"
        for r in agent_results
    ])

    synthesis_prompt = """You are a Travel Response Synthesizer. Combine multiple agent outputs into
a single, well-organized, comprehensive response.

RULES:
1. Organize logically (e.g., Flights → Hotels → Itinerary → Tips)
2. Don't repeat information
3. Highlight key recommendations
4. Note any conflicts or alternatives
5. Create clear sections with headers
6. End with actionable next steps

Original Query: {query}

Agent Results:
{results}

Create a unified, helpful response:"""

    response = llm.invoke([
        HumanMessage(content=synthesis_prompt.format(
            query=state["user_query"],
            results=results_formatted
        ))
    ])

    return {
        "final_answer": response.content,
        "messages": [AIMessage(content=response.content)]
    }

###################################
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

builder = StateGraph(TravelPlannerState)

builder.add_node('orchestrator', orchestrator)
builder.add_node('hotel_flight_agent_node', hotel_flight_agent_node)
builder.add_node('hotel_flight_tool_node', hotel_flight_tool_node)
builder.add_node('travel_scout_agent_node', travel_scout_agent_node)
builder.add_node('travel_scout_tool_node', travel_scout_tool_node)
builder.add_node('synthesizer', synthesizer_node)

builder.add_edge(START, 'orchestrator')
#orchestrator to hotel_flight_agent_node or travel_scout_agent_node
builder.add_conditional_edges('orchestrator', dispatch_to_agents, ['hotel_flight_agent_node', 'travel_scout_agent_node'])

#hotel_flight_agent_node to tool loop
builder.add_conditional_edges('hotel_flight_agent_node', should_continue_for_hotel_flight, ['hotel_flight_tool_node', 'synthesizer'])
builder.add_edge("hotel_flight_tool_node", "hotel_flight_agent_node") #tool loop
#travel_scout_agent_node to tool loop
builder.add_conditional_edges('travel_scout_agent_node', should_continue_for_travel_scout, ['travel_scout_tool_node', 'synthesizer'])
builder.add_edge("travel_scout_tool_node", "travel_scout_agent_node") #tool loop
#synthesizer to END
builder.add_edge( 'synthesizer', END)

thread = {'configurable': {'thread_id': '1'}}
agent = builder.compile(checkpointer = MemorySaver())

##############
##TESTING THE Graph
#############
result = agent.invoke({'user_query': 'Flights from Delhi to London on 2026-12-10 for 2 adults'}, config = thread)
print(result.get('final_answer'))

# result = agent.invoke({'user_query': 'Best time to visit Jodhpur?'}, config = thread)
##########################################
