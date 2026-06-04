from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, List, Annotated
import operator, os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv(r"D:/Learning/LangGraph_with_Claude/key.env")
api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)

class State(TypedDict):
    task: str
    plan: List[str]
    past_steps: Annotated[List, operator.add]
    final_response: str

# 1. PLANNER — creates the plan
def planner(state: State) -> dict:
    system = SystemMessage(content="""You are a planner. Break the task into 3-4 clear steps.
    Respond with ONLY a numbered list, one step per line. No extra text.""")
    response = llm.invoke([system, HumanMessage(content=state["task"])])
    
    # parse numbered list into list of steps
    steps = [line.strip() for line in response.content.split("\n") if line.strip()]
    print(f"📋 Plan created: {steps}")
    return {"plan": steps}

# 2. EXECUTOR — executes one step at a time
def executor(state: State) -> dict:
    # which step are we on?
    current_step_index = len(state["past_steps"])
    current_step = state["plan"][current_step_index]
    
    print(f"⚙️ Executing step {current_step_index + 1}: {current_step}")
    
    system = SystemMessage(content="You are an executor. Complete this step concisely.")
    response = llm.invoke([system, HumanMessage(content=current_step)])
    
    return {"past_steps": [(current_step, response.content)]}

# 3. ROUTER — more steps or done?
def should_continue(state: State) -> str:
    if len(state["past_steps"]) < len(state["plan"]):
        return "executor"   # more steps remaining
    return "compile"        # all done

# 4. COMPILER — combines all results
def compile_response(state: State) -> dict:
    all_results = "\n\n".join([f"{step}: {result}" for step, result in state["past_steps"]])
    system = SystemMessage(content="Compile these findings into a final answer.")
    response = llm.invoke([system, HumanMessage(content=all_results)])
    return {"final_response": response.content}

# Build graph
graph = StateGraph(State)
graph.add_node("planner", planner)
graph.add_node("executor", executor)
graph.add_node("compile", compile_response)

graph.add_edge(START, "planner")
graph.add_edge("planner", "executor")
graph.add_conditional_edges("executor", should_continue, {"executor": "executor", "compile": "compile"})
graph.add_edge("compile", END)

workflow = graph.compile()

# Run
result = workflow.invoke({"task": "Research the impact of AI on jobs", "past_steps": []})
print(f"\n✅ Final Response:\n{result['final_response']}")

# print(workflow.get_graph().draw_mermaid())
png = workflow.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png)
print("Graph saved as graph.png")