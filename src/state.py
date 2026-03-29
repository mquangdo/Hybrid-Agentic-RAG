import os, sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, TypedDict, Annotated
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
    """
    State definition for the ReAct agent.

    Attributes:
        messages: List of messages in the conversation (HumanMessage, AIMessage, ToolMessage)
    """

    messages: Annotated[List[BaseMessage], operator.add]
