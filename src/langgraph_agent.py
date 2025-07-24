# langgraph_agent.py

import os
from typing import List, Union
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langgraph.graph import StateGraph, END

from state_definition import GraphState
from pci_mock_logic import PCIMockLogic
from dotenv import load_dotenv  

# Load environment variables from .env file
load_dotenv()


class LangGraphAgent:
    def __init__(self, use_gemini: bool = True, memory_type: str = "buffer"):
        """
        Initializes the LangGraph agent for PCI query flow.
        """
        self.pci_logic = PCIMockLogic()

        # Initialize LLM
        if use_gemini:
            if not os.getenv("GOOGLE_API_KEY"):
                raise ValueError("GOOGLE_API_KEY environment variable not set.")
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite") # Use your verified model name
            print("Using Google Gemini API for LangGraph agent.")
        else:
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY environment variable not set.")
            self.llm = ChatOpenAI(temperature=0.7)
            print("Using OpenAI API for LangGraph agent.")

        # Initialize Langchain Memory (to be managed by the graph)
        if memory_type == "buffer":
            self.memory = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True
            )
            print("Using ConversationBufferMemory for LangGraph agent.")
        elif memory_type == "summary":
            self.memory = ConversationSummaryMemory(
                llm=self.llm, memory_key="chat_history", return_messages=True
            )
            print("Using ConversationSummaryMemory for LangGraph agent.")
        else:
            raise ValueError("Invalid memory_type. Choose 'buffer' or 'summary'.")

        self.workflow = StateGraph(GraphState)

        # --- Define Nodes ---

        def query_processing_node(state: GraphState) -> GraphState:
            """
            Node 1: Captures the query and prepares context for the LLM.
            Updates state with the current user query and loads chat history.
            """
            print("--- Executing: Query Processing Node ---")
            messages = state['messages']
            user_query = messages[-1].content # Last message is the current user input

            # Load chat history from Langchain memory
            chat_history_from_memory = self.memory.load_memory_variables({})["chat_history"]

            # Combine history with current query for the LLM to understand context
            # We'll pass this as part of the state for the next node if needed
            # For segmentation, we directly use the user_query and can also use chat_history_str
            
            # Update state with the user query
            state['user_query'] = user_query
            # Add historical messages to the state if they aren't already there implicitly
            # (LangGraph's state already handles message history if configured)
            # For PCILogic, we'll pass the chat history as a string.
            state['chat_history_str'] = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history_from_memory])
            return state

        def customer_segmentation_node(state: GraphState) -> GraphState:
            """
            Node 2: Runs basic mock customer segmentation logic.
            """
            print("--- Executing: Customer Segmentation Node ---")
            user_query = state['user_query']
            chat_history_str = state.get('chat_history_str', '') # Get history from state

            segment = self.pci_logic.get_customer_segment(user_query, chat_history_str)
            state['customer_segment'] = segment
            print(f"Detected Customer Segment: {segment}")
            return state

        def suggestion_node(state: GraphState) -> GraphState:
            """
            Node 3: Suggests a product, offer, or response based on the customer profile.
            """
            print("--- Executing: Suggestion Node ---")
            customer_segment = state['customer_segment']
            suggestion = self.pci_logic.get_suggestion(customer_segment)

            state['suggestion'] = suggestion

            # Generate LLM response combining the suggestion and recent history
            current_messages = state['messages']
            # Only include relevant history for the LLM prompt, excluding the current user message if it's already in current_messages
            # We pass the conversation history implicitly via the state's `messages`
            
            # The prompt now informs the LLM about the detected segment and the initial suggestion
            prompt_template = ChatPromptTemplate.from_messages([
                MessagesPlaceholder(variable_name="messages"), # This placeholder is crucial for LangGraph's message history
                ("system", f"You are a helpful customer service AI. The user's query has been analyzed, and their segment is '{customer_segment}'. A preliminary suggestion is: '{suggestion}'. Based on this, and the conversation history, formulate a friendly and helpful response. If the preliminary suggestion is sufficient, you can incorporate it directly. Otherwise, elaborate or rephrase to be most helpful to the user."),
            ])

            chain = prompt_template | self.llm | StrOutputParser()
            
            # The 'messages' in state already contains the HumanMessage that triggered this.
            # We need to add an AIMessage or similar to guide the LLM if needed for its internal thought process.
            # For simplicity, we just pass the existing messages to the prompt.
            # The chain takes the entire 'messages' list from the state directly.
            response = chain.invoke({"messages": current_messages, "customer_segment": customer_segment, "suggestion": suggestion})

            # Append AI's response to the messages for the next turn
            state['messages'].append(AIMessage(content=response))
            print(f"Generated Response: {response}")
            return state

        # --- Add Nodes to Workflow ---
        self.workflow.add_node("query_processing", query_processing_node)
        self.workflow.add_node("customer_segmentation", customer_segmentation_node)
        self.workflow.add_node("suggestion_generation", suggestion_node)

        # --- Set Entry and Exit Points ---
        self.workflow.set_entry_point("query_processing")
        self.workflow.add_edge("query_processing", "customer_segmentation")
        self.workflow.add_edge("customer_segmentation", "suggestion_generation")
        self.workflow.add_edge("suggestion_generation", END) # The graph ends here

        # --- Compile the Graph ---
        self.app = self.workflow.compile()


    def run_agent(self, user_input: str) -> str:
        """
        Runs the LangGraph agent with the given user input.
        Manages the Langchain memory manually for saving context.
        """
        # Load existing history from Langchain memory
        chat_history_from_memory = self.memory.load_memory_variables({})["chat_history"]
        
        # Initial state for LangGraph run
        initial_state = GraphState(
            messages=chat_history_from_memory + [HumanMessage(content=user_input)],
            user_query=user_input, # Store the immediate user query
            customer_segment="", # Will be updated by node
            suggestion="", # Will be updated by node
            error="" # Will be updated if error occurs
        )

        # Run the graph
        # The .stream() method yields the state at each step
        final_state = None
        for s in self.app.stream(initial_state):
            final_state = s
            # print(s) # Optional: print state at each step for debugging

        if final_state:
            # The last message in the 'messages' list of the final state is the agent's response
            ai_response = final_state['suggestion'] # Get the direct suggestion first
            
            # LangGraph's state `messages` should contain the AI response from suggestion_node
            # We want the *last* AI message, which is the final chatbot response.
            if final_state['messages']:
                last_message = final_state['messages'][-1]
                if isinstance(last_message, AIMessage):
                    ai_response = last_message.content

            # Save the interaction to Langchain memory
            self.memory.save_context(
                {"input": user_input},
                {"output": ai_response}
            )
            return ai_response
        else:
            return "An error occurred in the LangGraph agent."


# Example usage for testing (optional, for direct testing of this module)
if __name__ == "__main__":
    # Ensure your GOOGLE_API_KEY or OPENAI_API_KEY environment variable is set
    try:
        print("\n--- Testing LangGraph Agent (Gemini) ---")
        agent_gemini = LangGraphAgent(use_gemini=True, memory_type="buffer")

        print("\nUser: I want to know about discounts.")
        response = agent_gemini.run_agent("I want to know about discounts.")
        print("Agent:", response)

        print("\nUser: How do I upgrade my plan?")
        response = agent_gemini.run_agent("How do I upgrade my plan?")
        print("Agent:", response)

        print("\nUser: I'm really unhappy with the service, I'm thinking of leaving.")
        response = agent_gemini.run_agent("I'm really unhappy with the service, I'm thinking of leaving.")
        print("Agent:", response)

        print("\nUser: I'm a new user, how do I get started?")
        response = agent_gemini.run_agent("I'm a new user, how do I get started?")
        print("Agent:", response)

    except ValueError as e:
        print(f"Error: {e}. Please ensure your API key environment variable (GOOGLE_API_KEY or OPENAI_API_KEY) is set.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")