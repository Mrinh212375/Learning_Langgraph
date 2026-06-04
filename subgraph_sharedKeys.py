from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict,List, Annotated
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage 
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from langgraph.types import Send
import operator


load_dotenv(r"D:/Learning/LangGraph_with_Claude/key.env")
api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)
memory = MemorySaver()  

class ContentSupervisor(TypedDict):
    summary: str
    sentiment: str  
    messages: Annotated[List, operator.add]
    article: str

class DistributionSupervisor(TypedDict):
    summary: str
    sentiment: str  
    messages: Annotated[List, operator.add]
    tweet: str
    linkedin: str

class SupervisorState(TypedDict):
    article: str
    messages: Annotated[List, operator.add]
    summary: str
    sentiment: str
    tweet: str
    linkedin: str

##### Content Supervisor Graph #####

def dispatcher(state: ContentSupervisor) -> List[Send]:

    return [
            Send("summary_agent", {"article": state["article"]}),
            Send("sentiment_agent", {"article": state["article"]})
            ]


def summary_agent(state: ContentSupervisor) -> ContentSupervisor:

    system = SystemMessage(content="You are a helpful assistant that summarizes news articles.")
    query = state["article"]
    response = llm.invoke([system, HumanMessage(content=query)])
    return {"summary": response.content}

def sentiment_agent(state: ContentSupervisor) -> ContentSupervisor:

    system = SystemMessage(content="You are a helpful assistant that analyzes sentiment of news articles.")
    query = state["article"]
    response = llm.invoke([system, HumanMessage(content=query)])
    return {"sentiment": response.content}

contentgraph = StateGraph(ContentSupervisor)
contentgraph.add_node("summary_agent", summary_agent)
contentgraph.add_node("sentiment_agent", sentiment_agent)
contentgraph.add_conditional_edges(START, dispatcher)
contentgraph.add_edge("summary_agent", END)
contentgraph.add_edge("sentiment_agent", END)

contentflow = contentgraph.compile()

##### Distribution Supervisor Graph #####

def distribution_dispatcher(state: DistributionSupervisor) -> List[Send]:
    return [
            Send("tweet_agent", {"summary": state["summary"], "sentiment": state["sentiment"]}),
            Send("linkedin_agent", {"summary": state["summary"], "sentiment": state["sentiment"]})
            ]

def tweet_agent(state: DistributionSupervisor) -> DistributionSupervisor:

    system = SystemMessage(content="You are a helpful assistant that writes a tweet about a news article based on its summary and sentiment.")
    query = f"""Summary: {state["summary"]}\nSentiment: {state["sentiment"]}\nWrite a concise and engaging tweet about the news article based on the summary and sentiment."""
    response = llm.invoke([system, HumanMessage(content=query)])
    return {"tweet": response.content}

def linkedin_agent(state: DistributionSupervisor) -> DistributionSupervisor:

    system = SystemMessage(content="You are a helpful assistant that writes a LinkedIn post about a news article based on its summary and sentiment.")
    query = f"""Summary: {state["summary"]}\nSentiment: {state["sentiment"]}\nWrite a professional and engaging LinkedIn post about the news article based on the summary and sentiment."""
    response = llm.invoke([system, HumanMessage(content=query)])
    return {"linkedin": response.content}

distributiongraph = StateGraph(DistributionSupervisor)
distributiongraph.add_node("tweet_agent", tweet_agent)
distributiongraph.add_node("linkedin_agent", linkedin_agent)
distributiongraph.add_conditional_edges(START, distribution_dispatcher)
distributiongraph.add_edge("tweet_agent", END)
distributiongraph.add_edge("linkedin_agent", END)

distributionflow = distributiongraph.compile()


Supervisorgraph = StateGraph(SupervisorState)
Supervisorgraph.add_node("content_supervisor", contentflow)
Supervisorgraph.add_node("distribution_supervisor", distributionflow)
Supervisorgraph.add_edge(START, "content_supervisor")   
Supervisorgraph.add_edge("content_supervisor", "distribution_supervisor")
Supervisorgraph.add_edge("distribution_supervisor", END)

workflow = Supervisorgraph.compile()

if __name__ == "__main__":
    
    article = "The recent economic report shows a significant increase in consumer spending, indicating a strong recovery from the previous quarter's downturn. Experts are optimistic about the continued growth of the economy in the coming months."
    
    result = workflow.invoke({"article": article})
    print(f"Tweet: {result['tweet']}")
    print(f"LinkedIn: {result['linkedin']}")
    print(f"Summary: {result['summary']}")
    print(f"Sentiment: {result['sentiment']}")

    print(f"\n\n contentresult: {contentflow}")
    print(f"\n\n distributionresult: {distributionflow}")
    # print(f"\n\n result: {result}")
    print(workflow.get_graph().draw_ascii())