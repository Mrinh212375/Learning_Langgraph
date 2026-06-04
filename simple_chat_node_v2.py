from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import operator
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver

load_dotenv("D:\Learning\LangGraph_with_Claude\key.env")
api_key = os.getenv("GROQ_API_KEY")
#print(api_key)
llm = ChatGroq(model="llama-3.3-70b-versatile",api_key=api_key)
memory = MemorySaver()

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

workflow = graph.compile(checkpointer=memory)

# result = workflow.invoke({"messages":[HumanMessage("okay now tell me my name, age and where do I live")]})

# # print(result["response"])
# print(result)

result1 = workflow.invoke({"messages": [HumanMessage("My name is Mrinmoy")]},config = {"configurable": {"thread_id": "1"}})

# Turn 2 — pass previous messages + new message together
result2 = workflow.invoke({"messages": [HumanMessage("What is my name?")]}, config = {"configurable": {"thread_id": "1"}})

config2 = {"configurable": {"thread_id": "2"}}
result3 = workflow.invoke({"messages": [HumanMessage("What is my name?")]}, config=config2)

print(result2["messages"]) 
print(result3["messages"])


# Learnt: 

# 1. Simple graph workflow creation with one node
# 2. how to store memory manually in state of the graph using reducer - operator.add 
# 3. what is reducer actually 
# 4. How to invoke the graph manually to retain the memory from previous call.

# Learnt in v2:

#     1. Persistence in Langgraph using checkpointer.
#     2. Different Checkpointer availabale(RAM, SQLITE, POSTGRE) in Langgraph.checkpoint
#     3. "thread_id" is used as session concept, the histroy is attached with thread_id so that it will be reused later easily.