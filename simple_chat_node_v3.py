from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Union, Literal
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import operator
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage


load_dotenv("D:\Learning\LangGraph_with_Claude\key.env")
api_key = os.getenv("GROQ_API_KEY")
#print(api_key)

llm = ChatGroq(model="llama-3.3-70b-versatile",api_key=api_key)
memory = MemorySaver()

class State(TypedDict):
    # query: str
    # response: str
    messages: Annotated[list,operator.add]

@tool
def calculator(
    op: Literal["+", "-", "*", "/"],
    first: Union[int, float],
    second: Union[int, float],
) -> Union[int, float]:
    """
    Perform basic arithmetic on two numbers.

    Parameters:
    - op: Arithmetic operator.
      - "+" for addition (sum, add, plus)
      - "-" for subtraction (difference, minus, subtract)
      - "*" for multiplication (product, multiply, times)
      - "/" for division (divide, quotient)
    - first: First numeric operand (int or float).
    - second: Second numeric operand (int or float).

    Returns:
    - The computed numeric result.

    Raises:
    - ValueError: If an unsupported operator is provided.
    - ZeroDivisionError: If division by zero is attempted.
    """
    if op == "+":
        return first + second
    if op == "-":
        return first - second
    if op == "*":
        return first * second
    if op == "/":
        if second == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        return first / second

    raise ValueError(f"Unsupported operator: {op}")

@tool
def text_stats(text: str) -> str:
    """
    Return simple text statistics.
    Use this tool when the user asks about length/word count/character count.
    """
    chars = len(text)
    words = len(text.split())
    return f"chars={chars}, words={words}"


llm_with_tools = llm.bind_tools([calculator, text_stats])

def chatnode(state:State):

    system_message = SystemMessage(content="""You are a helpful assistant with access to a calculator tool.
    - ONLY use the calculator tool for math operations (add, subtract, multiply, divide)
    - For ALL other questions, answer directly from your knowledge
    - Never try to search the web or use tools that don't exist
    - If you feel you don't know the answer or certain tool is required to answer the question thne just simply return your thought briefly.ss
    """)

    query = [system_message] + state["messages"]
    print(f"\nQuery:\n {query}")
    response = llm_with_tools.invoke(query)
    if isinstance(response, AIMessage) and response.tool_calls:
        print(f"\nTool selected by LLM: {response.tool_calls[0]['name']}")
    print(f"\n Chatnode called: \nLLM Response:{response}")
    return {"messages":[response]}

def check_tool_calls(state:State):
    last_message = state["messages"][-1]
    # print(last_message)
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
    return END

tool_node = ToolNode([calculator, text_stats])
graph = StateGraph(State)

graph.add_node("chat_node",chatnode)
graph.add_node("tool_node",tool_node)
graph.add_edge(START,"chat_node")
graph.add_conditional_edges("chat_node",check_tool_calls)
graph.add_edge("tool_node","chat_node")
# graph.add_edge("chat_node",END)

workflow = graph.compile(checkpointer=memory)


result = workflow.invoke(
    {"messages":[HumanMessage("What is 12 multiplied by 7?")]},
    config={"configurable": {"thread_id": "1"}},
)
result2 = workflow.invoke(
    {"messages":[HumanMessage("Count words in: LangGraph makes tool calling easier")]},
    config={"configurable": {"thread_id": "2"}},
)

print(f"\n\nFinal Invoke Result:\n{result['messages'][-1].content}")
print(f"\n\nFinal Invoke Result 2:\n{result2['messages'][-1].content}")




# Learnt: 

# 1. Simple graph workflow creation with one node
# 2. how to store memory manually in state of the graph using reducer - operator.add 
# 3. what is reducer actually 
# 4. How to invoke the graph manually to retain the memory from previous call.

# Learnt in v2:

#     1. Persistence in Langgraph using checkpointer.
#     2. Different Checkpointer availabale(RAM, SQLITE, POSTGRE) in Langgraph.checkpoint
#     3. "thread_id" is used as session concept, the histroy is attached with thread_id so that it will be reused later easily.

# Learnt in v3:
# 1. Tools — how to create and bind tools to LLM
# 2. ToolNode — prebuilt node that executes tool calls
# 3. conditional_edge — routes between tool_node and END
# 4. Agent loop — chatnode called N+1 times (N = tool calls)
# 5. System message — guides LLM when to use tools vs answer directly
# 6. @tool decorator — what it adds vs without it
