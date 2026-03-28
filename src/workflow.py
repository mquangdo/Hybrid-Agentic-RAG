from typing import List, Any
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_core.tools import BaseTool
from state import AgentState


def build_workflow(llm_with_tools, tools: List[BaseTool]):
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

    def call_model(state: AgentState):
        """Node: Calls the LLM to generate response or tool calls."""
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def call_tools(state: AgentState):
        """Node: Executes tool calls."""
        tool_node = ToolNode(tools)
        tool_response = tool_node.invoke({"messages": state["messages"]})
        return tool_response

    def should_continue(state: AgentState):
        """Edge: Determines whether to continue with tools or end."""
        messages = state["messages"]
        last_message = messages[-1]

        if not last_message.tool_calls:
            return "end"
        return "tools"

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", call_tools)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "end": END}
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()
