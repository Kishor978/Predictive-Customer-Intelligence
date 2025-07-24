from typing import List
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict # Keep TypedDict for Python < 3.9 for direct usage

# Define the state for our LangGraph agent
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        messages: Conversation history (list of Langchain messages).
        user_query: The current raw user input.
        customer_segment: The determined customer segment (e.g., "high_value", "churn_risk").
        suggestion: The suggested product/offer/action.
        error: Any error message encountered during a step.
        chat_history_str: A string representation of the full chat history for PCI logic.
    """
    messages: List[BaseMessage] # Simpler definition, ensure initial_state always provides a list
    user_query: str
    customer_segment: str
    suggestion: str
    error: str
    chat_history_str: str # Added this field