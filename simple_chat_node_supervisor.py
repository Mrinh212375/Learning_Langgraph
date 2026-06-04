from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated, Union, Literal
import os
import operator
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool


load_dotenv(r"D:/Learning/LangGraph_with_Claude/key.env")
api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)
memory = MemorySaver()


class State(TypedDict, total=False):
    messages: Annotated[list, operator.add]


@tool
def calculator(
    op: Literal["+", "-", "*", "/"],
    first: Union[int, float],
    second: Union[int, float],
) -> Union[int, float]:
    """Perform basic arithmetic on two numbers."""
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


def run_math_agent(query: str) -> str:
    """Run a small inner tool loop so the math agent can call calculator and return a final answer."""
    system = SystemMessage(
        content=(
            "You are a math expert. Use the calculator tool for arithmetic. "
            "If numbers are ambiguous, ask a clarification question."
        )
    )

    messages = [system, HumanMessage(content=query)]

    for _ in range(5):
        response = math_llm.invoke(messages)
        messages.append(response)

        if not isinstance(response, AIMessage) or not response.tool_calls:
            return response.content

        for call in response.tool_calls:
            if call["name"] != "calculator":
                continue
            result = calculator.invoke(call["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    return "I could not complete the calculation in time. Please try again."



@tool
def general_agent_tool(query: str) -> str:
    """Answer non-math/general questions."""
    system = SystemMessage(content="You are a helpful assistant for general questions.")
    response = llm.invoke([system, HumanMessage(content=query)])
    return response.content


@tool
def math_agent_tool(query: str) -> str:
    """Answer arithmetic questions by using calculator when needed."""
    return run_math_agent(query)


supervisor_llm = llm.bind_tools([general_agent_tool, math_agent_tool])


def supervisor_node(state: State):
    system = SystemMessage(
        content=(
            "You are a supervisor with two tools: "
            "1) general_agent_tool for general Q&A "
            "2) math_agent_tool for arithmetic/math. "
            "Choose and call the correct tool when needed, then provide a final concise answer."
        )
    )
    response = supervisor_llm.invoke([system] + state["messages"])
    print(f"\nSupervisor response:\n{response}")
    return {"messages": [response]}


supervisor_tool_node = ToolNode([general_agent_tool, math_agent_tool])


def route_from_supervisor(state: State):
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "supervisor_tool_node"
    return END


graph = StateGraph(State)

graph.add_node("supervisor_node", supervisor_node)
graph.add_node("supervisor_tool_node", supervisor_tool_node)

graph.add_edge(START, "supervisor_node")
graph.add_conditional_edges(
    "supervisor_node",
    route_from_supervisor,
    {"supervisor_tool_node": "supervisor_tool_node", END: END},
)
graph.add_edge("supervisor_tool_node", "supervisor_node")

workflow = graph.compile(checkpointer=memory)


result = workflow.invoke(
    {"messages": [HumanMessage("What is 347 multiplied by 829?")]},
    config={"configurable": {"thread_id": "3"}},
)
print(f"Final state: {result}")
print(f"\nFinal Invoke Result:\n{result['messages'][-1].content}")
