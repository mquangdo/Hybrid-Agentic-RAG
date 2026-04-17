import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langfuse.langchain import CallbackHandler
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph
from operator import add as add_messages
load_dotenv()


langfuse_handler = CallbackHandler()

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7, api_key=os.getenv("GROQ_API_KEY"))
# prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
# chain = prompt | llm
# print(chain.invoke({"topic": "France"}, config={"callbacks": [langfuse_handler]}))

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# The chatbot node function takes the current State as input and returns an updated messages list. This is the basic pattern for all LangGraph node functions.
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# Add a "chatbot" node. Nodes represent units of work. They are typically regular python functions.
graph_builder.add_node("chatbot", chatbot)

# Add an entry point. This tells our graph where to start its work each time we run it.
graph_builder.set_entry_point("chatbot")

# Set a finish point. This instructs the graph "any time this node is run, you can exit."
graph_builder.set_finish_point("chatbot")

# To be able to run our graph, call "compile()" on the graph builder. This creates a "CompiledGraph" we can use invoke on our state.
graph = graph_builder.compile()

graph.invoke({"messages": [HumanMessage(content="Tell me a joke about France")]}, config={"callbacks": [langfuse_handler]})