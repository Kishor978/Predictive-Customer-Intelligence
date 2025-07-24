# state_definition.py

from typing import List, Annotated, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing_extensions import TypedDict
from langchain_core.pydantic_v1 import BaseModel, Field

# Define the state for our LangGraph agent
# This will hold the conversation history and other dynamic data
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        messages: Conversation history (list of Langchain messages).
        user_query: The current raw user input.
        customer_segment: The determined customer segment (e.g., "high_value", "churn_risk").
        suggestion: The suggested product/offer/action.
        error: Any error message encountered during a step.
    """
    messages: Annotated[List[BaseMessage], Field(default_factory=list)]
    user_query: str
    customer_segment: str
    suggestion: str
    error: str