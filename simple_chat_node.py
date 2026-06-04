from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import operator
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv("D:\Learning\LangGraph_with_Claude\key.env")
api_key = os.getenv("GROQ_API_KEY")
#print(api_key)
llm = ChatGroq(model="llama-3.3-70b-versatile",api_key=api_key)

class State(TypedDict):
    # query: str
    # response: str
    messages: Annotated[list,operator.add]

def chatnode(state:State):

    query = state["messages"]
    response = llm.invoke(query)
    return {"messages":[AIMessage(response.content)]}

graph = StateGraph(State)

graph.add_node("chat_node",chatnode)

graph.add_edge(START,"chat_node")
graph.add_edge("chat_node",END)

workflow = graph.compile()

# result = workflow.invoke({"messages":[HumanMessage("okay now tell me my name, age and where do I live")]})

# # print(result["response"])
# print(result)

result1 = workflow.invoke({"messages": [HumanMessage("My name is Mrinmoy")]})

# Turn 2 — pass previous messages + new message together
result2 = workflow.invoke({"messages": result1["messages"] + [HumanMessage("What is my name?")]})

print(result2["messages"]) 


# Learnt: 

# 1. Simple graph workflow creation with one node
# 2. how to store memory manually in state of the graph using reducer - operator.add 
# 3. what is reducer actually 
# 4. How to invoke the graph manually to retain the memory from previous call.