from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

load_dotenv(r"D:/Learning/LangGraph_with_Claude/key.env")
api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)
memory = MemorySaver()

class State(TypedDict):
    article: str
    summary: str
    sentiment: str
    tweet: str
    feedback: str


def summary_agent(state: State) -> State:
    system = SystemMessage(content="You are a helpful assistant that summarizes news articles.")
    query = state["article"]
    response = llm.invoke([system, HumanMessage(content=query)])
    return {"summary": response.content}


def sentiment_agent(state: State) -> State:
    system = SystemMessage(content="You are a helpful assistant that analyzes sentiment of news articles from its summary.")
    query = state["summary"]
    response = llm.invoke([system, HumanMessage(content=query)])
    return {"sentiment": response.content}


def tweet_agent(state: State) -> State:

    if state.get("feedback"):
        feedback_msg = f"The previous generated tweet received the following feedback: {state['feedback']}. Please use this feedback to improve the last tweet: {state['tweet']}"
        query = f"""Summary: {state["summary"]}\nSentiment: {state["sentiment"]}\nImprove the last tweet about the news article based on the summary, sentiment and the feedback."""
        response = llm.invoke(feedback_msg + query)
        return {"tweet": response.content}
    
    system = "You are a helpful assistant that writes a tweet about a news article based on its summary and sentiment."
    query = f"""Summary: {state["summary"]}\nSentiment: {state["sentiment"]}\nWrite a concise and engaging tweet about the news article based on the summary and sentiment."""
    response = llm.invoke(system + query)
    return {"tweet": response.content}


def human_review(state: State) -> Command:

    decision = interrupt({
        "tweet": state["tweet"],
        "action": "Approve or provide feedback for improvement of the generated tweet."
    })
    print("Human Review Decision:", decision)

    approved = decision.get("approved")
    if approved == 'yes':
        return Command(goto=END) 

    feedback = decision.get("feedback", "")
    return Command(update={"feedback": feedback}, goto="tweet_agent")

graph = StateGraph(State)
graph.add_node("summary_agent", summary_agent)
graph.add_node("sentiment_agent", sentiment_agent)  
graph.add_node("tweet_agent", tweet_agent)
graph.add_node("human_review", human_review)
graph.add_edge(START, "summary_agent")
graph.add_edge("summary_agent", "sentiment_agent")  
graph.add_edge("sentiment_agent", "tweet_agent")
graph.add_edge("tweet_agent", "human_review")
# graph.add_edge("human_review", END)

workflow = graph.compile(checkpointer=memory)

if __name__ == "__main__":

    article = """
    The United States military has launched a blockade of all Iranian ports
    starting Monday April 13, 2026 at 10 AM ET, following the collapse of
    peace talks in Islamabad, Pakistan over the weekend.

    Vice President JD Vance, who led the US delegation, confirmed that
    marathon negotiations lasting over 21 hours ended without agreement.
    "The bad news is that we have not reached an agreement. They have chosen
    not to accept our terms," Vance told reporters before departing Pakistan.

    President Trump announced the US Navy will prevent ships from passing
    through the Strait of Hormuz, through which a fifth of global crude supply
    normally passes. Oil prices surged above $100 a barrel following the
    announcement, rattling Asian markets.

    Iran's Islamic Revolutionary Guard Corps warned that any military vessels
    approaching the Strait of Hormuz "will be dealt with harshly and
    decisively." The blockade applies to all vessels entering or departing
    Iranian ports, but will not affect ships transiting to non-Iranian ports.

    More than 5,000 people have been killed since the conflict began six weeks
    ago, including at least 1,701 civilians in Iran and 2,089 in Lebanon.
    """

    config = {"configurable": {"thread_id": "1"}}
    result = workflow.invoke({"article": article}, config=config)

    while True:
        print(f"The generated tweet:{result['__interrupt__'][0].value['tweet']}")
        user_input = input("\nDo you approve the tweet? (yes/no): ")
        feedback = ""
        if user_input.lower() not in ["yes", "no"]:
            print("Invalid input. Please enter 'yes' or 'no'.")
            continue

        if user_input.lower() == "no":
            feedback = input("Please provide feedback for improvement: ")
        #     revised = workflow.invoke(
        #         Command(resume={"approved": False, "feedback": feedback}),
        #         config=config,
        #     )
        # else:
        #     revised = workflow.invoke(
        #         Command(resume={"approved": False, "feedback": "make it shorter"}),
        #     config=config,
        # )
        # print(revised["__interrupt__"])
        result = workflow.invoke(Command(resume={"approved": user_input.lower(), "feedback": feedback}), config=config)
        if result.get("__interrupt__") is None:
            break

    # print(result["tweet"])
    # print("--- Full Result ---")
    # print(result)

    print(result["tweet"])
    print("--- Revised Result ---")
    print(result)

    # print(final["tweet"])
    # print("--- Final Result ---")
    # print(final)


# Learnt in HITL:
# 1. interrupt() — pauses graph, saves state, returns immediately
# 2. Command(resume=...) — resumes paused graph
# 3. __interrupt__ — None means graph reached END
# 4. while True loop — handles multiple human rejections
# 5. thread_id — bridges multiple invoke() calls
# 6. Production = 2 API endpoints, Learning = manual invoke()
# 7. llm.invoke() always needs list of message objects