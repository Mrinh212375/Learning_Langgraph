from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated, Union, Literal
import os
import operator
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool


load_dotenv(r"D:/Learning/LangGraph_with_Claude/key.env")
api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)
memory = MemorySaver()


class State(TypedDict, total=False):
    messages: Annotated[list, operator.add]
    route: str


@tool
def calculator(
    op: Literal["+", "-", "*", "/"],
    first: Union[int, float],
    second: Union[int, float],
) -> Union[int, float]:
    """
    Perform basic arithmetic on two numbers.
    op:
      "+" for addition (sum, add, plus)
      "-" for subtraction (difference, minus, subtract)
      "*" for multiplication (product, multiply, times)
      "/" for division (divide, quotient)
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


math_llm = llm.bind_tools([calculator])
calculator_tool_node = ToolNode([calculator])


def supervisor_node(state: State):
    system = SystemMessage(
        content=(
            "You are a router. Decide if user intent is arithmetic math or general Q&A. "
            "Respond with exactly one word: MATH or GENERAL."
        )
    )
    response = llm.invoke([system] + state["messages"])
    route = "math_agent_node" if "MATH" in response.content.upper() else "general_agent_node"
    print(f"\nSupervisor route: {route}")
    return {"route": route}


def route_from_supervisor(state: State):
    return state.get("route", "general_agent_node")


def general_agent_node(state: State):
    system = SystemMessage(content="You are a helpful assistant for general questions.")
    response = llm.invoke([system] + state["messages"])
    print(f"\nGeneral agent response:\n{response}")
    return {"messages": [response]}


def math_agent_node(state: State):
    system = SystemMessage(
        content=(
            "You are a math assistant with calculator tool access. "
            "Use calculator only when both operands are explicit numbers. "
            "If numbers are ambiguous, ask a clarification question and do not call tools."
        )
    )
    response = math_llm.invoke([system] + state["messages"])
    print(f"\nMath agent response:\n{response}")
    return {"messages": [response]}


def route_from_math_agent(state: State):
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "calculator_tool_node"
    return END


graph = StateGraph(State)

graph.add_node("supervisor_node", supervisor_node)
graph.add_node("general_agent_node", general_agent_node)
graph.add_node("math_agent_node", math_agent_node)
graph.add_node("calculator_tool_node", calculator_tool_node)

graph.add_edge(START, "supervisor_node")
graph.add_conditional_edges(
    "supervisor_node",
    route_from_supervisor,
    {
        "math_agent_node": "math_agent_node",
        "general_agent_node": "general_agent_node",
    },
)
graph.add_edge("general_agent_node", END)
graph.add_conditional_edges(
    "math_agent_node",
    route_from_math_agent,
    {"calculator_tool_node": "calculator_tool_node", END: END},
)
graph.add_edge("calculator_tool_node", "math_agent_node")

workflow = graph.compile(checkpointer=memory)

result = workflow.invoke(
    {"messages": [HumanMessage("What is 347 multiplied by 829?")]},
    config={"configurable": {"thread_id": "3"}},
)
print(f"Final state:{result}")
print(f"\n\nFinal Invoke Result:\n{result['messages'][-1].content}")


### Learnt:
# 1. How to create multiple nodes in a graph and route between them based on LLM decisions.
# 2. How to use a supervisor node to route between different agents (math vs general Q&A).
# 3. How to conditionally route back to the math agent after a tool call if the math agent's response included a tool call.
# 4. How to structure the state to include both messages and routing information for flexible graph workflows.  
# 5. Moreover this is a Routing Multiagent patter, not the actual Supervisor Agent pattern. In Supervisor Agent pattern, the supervisor agent would also have access to the tools and would be responsible for calling them directly instead of routing to a separate tool node.