from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import operator, os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv(r"D:\Learning\Learning_Langgraph\key.env")
api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)

class State(TypedDict):
    messages: Annotated[list, operator.add]

def chat_node(state: State, *, store: BaseStore):
    user_id = "user_123"
    
    # 1. Read memories from store
    memories = store.search((user_id, "memories"))
    print(f"Memory anatomy: {memories}")
    memory_text = "\n".join([m.value["data"] for m in memories]) if memories else "Nothing yet"
    
    # 2. Use in system prompt
    system = SystemMessage(content=f"""You are a helpful assistant.
    What you remember about this user:
    {memory_text}""")
    
    # 3. Get response
    response = llm.invoke([system] + state["messages"])
    
    # 4. Save the user's message as a memory
    last_user_msg = state["messages"][-1].content
    store.put((user_id, "memories"), f"mem_{len(memories)}", {"data": last_user_msg})
    
    return {"messages": [response]}

# Build graph
graph = StateGraph(State)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

checkpointer = MemorySaver()
store = InMemoryStore()
workflow = graph.compile(checkpointer=checkpointer, store=store)

# Test across DIFFERENT threads!
config1 = {"configurable": {"thread_id": "1"}}
workflow.invoke({"messages": [HumanMessage("My name is Mrinmoy and I love LangGraph")]}, config=config1)

# Different thread — but should still remember!
config2 = {"configurable": {"thread_id": "2"}}
result = workflow.invoke({"messages": [HumanMessage("What do you know about me?")]}, config=config2)
print(result["messages"][-1].content)