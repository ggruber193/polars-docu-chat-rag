from typing import TypedDict, Any, List
import os
from functools import partial

from langgraph.constants import END
from qdrant_client import QdrantClient

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.rate_limiters import InMemoryRateLimiter, BaseRateLimiter
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage

from langgraph.graph import START, StateGraph, Graph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from src.database.qdrant_store import QdrantStore
from src.embeddings import TextEmbedder
from src.config import EMBEDDING_MODEL, QDRANT_COLLECTION_NAME, CHAT_API_KEY, QDRANT_URL, QDRANT_API_KEY

RAG_PROMPT_STR = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
\n
{context} 
"""
RAG_PROMPT = PromptTemplate.from_template(RAG_PROMPT_STR)

embedding_model = TextEmbedder(modelname=EMBEDDING_MODEL)

client = QdrantClient(QDRANT_URL, api_key=QDRANT_API_KEY)
qdrant_store = QdrantStore(client)

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.25,  # <-- Super slow! We can only make a request once every 10 seconds!!
    check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
    max_bucket_size=15,  # Controls the maximum burst size.
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    google_api_key=CHAT_API_KEY,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# init_chat_model("google_vertexai:gemini-2.0-flash", rate_limiter=rate_limiter, )


class State(TypedDict):
    question: str
    context: List[str]
    answer: str


def query_or_respond(state: MessagesState):
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


@tool
def retrieve(query: str):
    """Retrieve information related to a query, specific to the python polars package"""
    retrieved_docs = []
    if qdrant_store is not None:
        query = embedding_model.embed_text(query)
        retrieved_docs = qdrant_store.get_topk_points_single(query[0], QDRANT_COLLECTION_NAME, k=5)
    else:
        retrieved_docs = []
    return '\n\n'.join(retrieved_docs)


def generate(state: MessagesState):
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]
    system_message_content = RAG_PROMPT_STR.format(context=tool_messages)
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
           or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    response = llm.invoke(prompt)
    return {"messages": [response]}


tools = ToolNode([retrieve])

graph_builder = StateGraph(MessagesState)

graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

if __name__ == '__main__':
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": "def234"}}

    user_input = "Hi there! My name is Will."

    # The config is the **second positional argument** to stream() or invoke()!
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )
    for event in events:
        event["messages"][-1].pretty_print()

    print(graph.get_state(config))
    print(memory.get(config))

    user_input = "Remember my name?"
    config = {"configurable": {"thread_id": "def234"}}

    # The config is the **second positional argument** to stream() or invoke()!
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )
    for event in events:
        event["messages"][-1].pretty_print()

    print(graph.get_state(config))
    print(memory.get(config))

    user_input = "Remember my name?"
    config = {"configurable": {"thread_id": "ddef234"}}

    # The config is the **second positional argument** to stream() or invoke()!
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )
    for event in events:
        event["messages"][-1].pretty_print()

