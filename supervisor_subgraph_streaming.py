# from pyexpat.errors import messages

from pyexpat.errors import messages

from hamcrest import instance_of
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from sympy import content
from typing_extensions import TypedDict,List, Annotated 
import operator
import os
from langchain_groq import ChatGroq 
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

load_dotenv(r"D:/Learning/Learning_Langgraph/key.env")
api_key = os.getenv("GROQ_API_KEY")
print(f"apikey: {api_key}")
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)



##### Flight Subgraph #####

class flightState(TypedDict):
    # origin: str
    # destination: str
    # date: str
    # flight_options: Annotated[List, operator.add]
    # flight_type: str
    messages: Annotated[List, operator.add]

def domestic_agent(state: flightState) -> flightState:

    system = SystemMessage(content="You are a helpful assistant that finds domestic flight options based on the user's origin, destination and date.")
    query = f"Find only flight options by looking at this user query - {state['messages'][0].content}."
    response = llm.invoke([system, HumanMessage(content=query)])
    return {"messages": [AIMessage(content = response.content)]}

def international_agent(state: flightState) -> flightState:

    system = SystemMessage(content="You are a helpful assistant that finds international flight options based on the user's origin, destination and date.")
    query = f"Find only flight options by looking at this user query - {state['messages'][0].content}."
    response = llm.invoke([system, HumanMessage(content=query)])
    return {"messages": [AIMessage(content = response.content)]}


def flightsupervisor_agent(state: flightState) -> Command:

    system = SystemMessage(content = "You are a helpful assistant that determines whether the user's flight request is domestic or international based on the origin and destination.")
    query = f"""Determine if the flight is domestic or international by looking at this user query - {state['messages'][0].content}. Just respond with either 'domestic' or 'international'."""
    response = llm.invoke([system, HumanMessage(content=query)])
    # print("Supervisor Response:", response.content)
    if "domestic" in response.content.lower():

        return Command(goto="domestic_agent",update={"messages": [AIMessage(content=response.content)]})
    else:
        return Command(goto="international_agent",update={"messages": [AIMessage(content=response.content)]})

flightgraph = StateGraph(flightState)
flightgraph.add_node("flightsupervisor", flightsupervisor_agent)
flightgraph.add_node("domestic_agent", domestic_agent)
flightgraph.add_node("international_agent", international_agent)

flightgraph.add_edge(START, "flightsupervisor")
# flightgraph.add_conditional_edges("flightsupervisor",)
flightgraph.add_edge("domestic_agent", END)
flightgraph.add_edge("international_agent", END)
flight_workflow = flightgraph.compile()


#### Hotel Subgraph #####

class hotelState(TypedDict):
    # location: str
    # date: str
    # hotel_options: Annotated[List, operator.add]
    # hotel_type: str
    messages: Annotated[List, operator.add]

@tool
def find_budget_hotels(location: str, date: str) -> str:

    """this tool helps to findout budget hotel options based on location, date"""

    print(f"\n\nFinding budget hotels in {location} on {date}...")
    query = f"Your are an expert finding budget hotels in any location.Find hotel options in {location} on {date}.Just respond with the hotel options without any additional text."
    response = llm.invoke([HumanMessage(content=query)])
    return response.content

@tool
def find_luxury_hotels(location: str, date: str) -> str:

    """this tool helps to findout luxury hotel options based on location, date."""

    print(f"\n\nFinding luxury hotels in {location} on {date}...")
    query = f"Your are an expert finding luxury hotels in any location.Find hotel options in {location} on {date}.Just respond with the hotel options without any additional text."
    response = llm.invoke([HumanMessage(content=query)])
    return response.content

llm_with_tool = llm.bind_tools([find_budget_hotels, find_luxury_hotels])
# def budget_hotel_agent(state: hotelState) -> hotelState:

#     system = SystemMessage(content="You are a helpful assistant that finds budget hotel options based on the user's location and date.")
#     query = f"Find budget hotel options in {state['location']} on {state['date']}."
#     response = llm.invoke([system, HumanMessage(content=query)])
#     return {"hotel_options": [response.content]}

# def luxury_hotel_agent(state: hotelState) -> hotelState:

#     system = SystemMessage(content="You are a helpful assistant that finds luxury hotel options based on the user's location and date.")
#     query = f"Find luxury hotel options in {state['location']} on {state['date']}."
#     response = llm.invoke([system, HumanMessage(content=query)])
#     return {"hotel_options": [response.content]}
toolnode = ToolNode([find_budget_hotels, find_luxury_hotels])

def hotelsearch_agent(state: hotelState) -> dict:

    system = SystemMessage(content = "You are a supervisor having access to two tools: find_budget_hotels and find_luxury_hotels. Based on the user's preferences, determine which tool to use to find hotel options.")
    # query = f"User is looking for a {state['hotel_type']} hotel in {state['location']} on {state['date']}."
    response = llm_with_tool.invoke([system, state["messages"][0]])
    # print("Hotel Search Agent Response:", response)
    return {'messages': [response]}
    # hotel_options = []
    # if isinstance(response,AIMessage) and response.tool_calls:
    #     tool_call = response.tool_calls[0]
    #     print(f"Tool call details: {tool_call}")
    #     tool_name =  tool_call.get("name")
    #     print(tool_name)
    #     if tool_name == "find_budget_hotels":
    #         hotel_options.append(find_budget_hotels.invoke({"location": state["location"], "date": state["date"]}))
    #     elif tool_name == "find_luxury_hotels":
    #         hotel_options.append(find_luxury_hotels.invoke({"location": state["location"], "date": state["date"]}))

        # return {"hotel_options": hotel_options}

hotelgraph = StateGraph(hotelState)
hotelgraph.add_node("hotelsearch_agent", hotelsearch_agent)
hotelgraph.add_node("toolnode", toolnode)
hotelgraph.add_edge(START, "hotelsearch_agent")
hotelgraph.add_edge("hotelsearch_agent", "toolnode")
hotelgraph.add_edge("toolnode",END)
hotel_workflow = hotelgraph.compile()

#### top supervisor agent #####

class topSupervisorState(TypedDict):
    messages: Annotated[List, operator.add]
    # top_supervisor_response: str
    # origin: str
    # destination: str
    # date: str
    # location: str
    # hotel_type: str
    # flight_options: Annotated[List, operator.add]
    # hotel_options: Annotated[List, operator.add]

def top_supervisor_agent(state: topSupervisorState) -> topSupervisorState:

    # system = SystemMessage(content =       
    #                        '''
    #                         Your are a supervisor agent, supervising two other agent - flightagent & hotelagent. 
    #                         your task is to analyse the messages from beginning to end and decide the correct routing each and every time.
    #                         The very first message in the list is the user message, see all the messages appended till now and decide the correct route now.
    #                             - once flightagent workdone is over, see the user message for any requirement for hotel also, If so then route to hotelagent.
    #                             - For only flightagent requirement in user query just route to end after flightagnet workdone.
    #                             - For only hotelagent requirement in user query just route to end after hotelagent workdone.
    #                         '''
    #                         )
    system = SystemMessage(content="""You are a supervisor routing between two agents:
                                        - 'flights' agent: handles flight bookings
                                        - 'hotels' agent: handles hotel bookings

                                        CRITICAL RULES:
                                        1. The FIRST message in the list is the user's original request — analyze it carefully.
                                        2. Check what user asked for: flight only? hotel only? both?
                                        3. Look at remaining messages to see what's ALREADY DONE.
                                        4. Route to whichever is STILL PENDING.
                                        5. If everything user asked is done — respond 'end'.

                                        Respond with ONLY ONE WORD: 'flights', 'hotels', or 'end'. Nothing else.
                                        """)
    query = f"""analyse the messages provided here: {state['messages']}. you have to route among 'flights', 'hotels' & 'end'. Just respond your routing decision in single word, nothing else."""
    response = llm.invoke([system, HumanMessage(content=query)])
    # print("Top Supervisor Response:", response.content)
    return {"messages": [AIMessage(content=response.content)]}
    
def top_supervisor_router(state:topSupervisorState) -> str:
    if state["messages"][-1].content.strip().lower() == 'flights':
        return "flightagent"
    elif state["messages"][-1].content.strip().lower() == 'hotels':
        return "hotelagent"
    elif state["messages"][-1].content.strip().lower() == 'end':

        return "End"

maingraph = StateGraph(topSupervisorState)
maingraph.add_node("topsupervisor",top_supervisor_agent)
maingraph.add_node("flightagent", flight_workflow)
maingraph.add_node("hotelagent",hotel_workflow)

maingraph.add_edge(START,"topsupervisor")
maingraph.add_conditional_edges("topsupervisor",top_supervisor_router,{"flightagent":"flightagent","hotelagent":"hotelagent","End":END})
maingraph.add_edge("flightagent","topsupervisor")
maingraph.add_edge("hotelagent","topsupervisor")
mainworkflow = maingraph.compile()




if __name__ == "__main__":

    # result = flight_workflow.invoke({"origin": "Mumbai",
    #                                 "destination": "London",
    #                                 "date": "2024-12-15"}
    #                                 )
    # print(f"Flight Options: {result['flight_options']}")
    # print(flight_workflow.get_graph().draw_ascii())

    # result = hotelgraph.compile().invoke({"location": "Paris",
    #                                     "date": "2024-12-20",   
    #                                     "hotel_type": "luxury"}
    #                                     )
    # print(f"Hotel Options: {result['hotel_options']}")
    # print(hotel_workflow.get_graph().draw_ascii())
    # result = mainworkflow.invoke({"messages":[HumanMessage(content = "Find some flights from kolkata to Goa for 24th June ans also some luxury hotels in Goa for the same date.")]})
    for chunk in mainworkflow.stream({"messages":[HumanMessage(content = "Find some flights from kolkata to Goa for 24th June ans also some luxury hotels in Goa for the same date.")]},stream_mode="messages"):
        # print(f"len of messages now: {len(chunk['messages'])}")
        print(f"Type of chunk:{instance_of(type(chunk[0]))} and chunk content:\n{chunk[0].content}")
        # print(chunk)

    # print(f"final result:{result['messages'][-1].content}")
    # print(result)
    # print(mainworkflow.get_graph().draw_ascii())
# please find me flights from Kolkata to Goa on 24th June and also some budget hotels in GOA for 24th June



''' Points to remember:

    1.many streaming modes are there - upadtes, messages, values, debug, tasks etc. by default is updates I guess.
    2.straming() can yield output without waiting till the last of the execution unlike invoke()

'''