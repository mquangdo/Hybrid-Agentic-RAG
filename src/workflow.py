import os, sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Any
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_core.tools import BaseTool
from langfuse.langchain import CallbackHandler
from state import AgentState
from langchain_groq import ChatGroq
from tools import get_tools
from dotenv import load_dotenv
from functools import lru_cache

load_dotenv()  

#Set up functions
@lru_cache(maxsize=1)
def get_llm(have_tools: bool = True) -> ChatGroq:
    """
    Initializes and returns the LLM model with tools bound.

    This function sets up the ChatGroq model with the appropriate API key
    and binds the document retriever tool to it for use in the workflow.

    Args:
        None
        
    Returns:
        An instance of the LLM model with tools bound for invocation in the workflow
    """
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7, api_key=os.getenv("GROQ_API_KEY"))
    tools = get_tools()
    llm_with_tools = llm.bind_tools(tools)
    return llm_with_tools


def call_model(state: AgentState):
        """Node: Calls the LLM to generate response or tool calls."""
        messages = state["messages"]
        
        llm_with_tools = get_llm()
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}


def should_continue(state: AgentState):
    """Edge: Determines whether to continue with tools or end."""
    messages = state["messages"]
    last_message = messages[-1]

    if not last_message.tool_calls:
        return "end"
    return "tools"


def build_workflow():
    """
    Builds a StateGraph workflow with ReAct pattern.

    Creates a graph with nodes for the agent (LLM) and tools, with
    conditional edges that implement the ReAct pattern:
    1. Agent generates response with potential tool calls
    2. If tool calls exist, route to tools node
    3. Tools execute and return observations
    4. Agent observes results and either continues or finishes

    Args:
        llm_with_tools: LLM model with tools bound to it
        tools: List of tools to use in the workflow

    Returns:
        Compiled StateGraph workflow
    """

    
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools=get_tools()))

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "end": END}
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()

if __name__ == "__main__":
    workflow = build_workflow()
    # Example input to start the workflow
    initial_state = {"messages": [HumanMessage(content="What is normalization, what is attention?")]}
    result = workflow.invoke(initial_state, config={"callbacks": [CallbackHandler()]})
    print(result["messages"][-1].content)
    
    import os
    # Giả sử bạn đã định nghĩa workflow của mình
    # app = workflow.compile()

    # Thư mục để lưu ảnh (tùy chọn)
    output_dir = "images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, "graph.png")

    # Lấy dữ liệu ảnh PNG từ graph
    # LangGraph sử dụng Mermaid.ink API để render ảnh online
    try:
        # draw_mermaid_png() trả về dữ liệu binary của ảnh
        png_data = workflow.get_graph().draw_mermaid_png()
        
        # Ghi dữ liệu binary vào file
        with open(file_path, "wb") as f:
            f.write(png_data)
        print(f"✅ Đã lưu ảnh graph thành công tại: {file_path}")
        
    except Exception as e:
        print(f"❌ Lỗi khi lưu ảnh: {e}")
        print("Có thể do vấn đề kết nối mạng (để gọi Mermaid API) hoặc thiếu thư viện bổ trợ.")


