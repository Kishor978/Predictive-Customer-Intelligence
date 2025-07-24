import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langgraph.graph import StateGraph, END

from src.state_definition import GraphState
from src.pci_mock_logic import PCIMockLogic

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

            # Load chat history from Langchain memory (this memory is separate from graph state messages)
            chat_history_from_memory = self.memory.load_memory_variables({})["chat_history"]

            # Combine history into a string for PCI logic if needed
            chat_history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history_from_memory])
            
            state['user_query'] = user_query
            state['chat_history_str'] = chat_history_str # Store as string in state
            return state

        def customer_segmentation_node(state: GraphState) -> GraphState:
            """
            Node 2: Runs basic mock customer segmentation logic.
            """
            print("--- Executing: Customer Segmentation Node ---")
            user_query = state['user_query']
            chat_history_str = state['chat_history_str'] # Get history from state

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
            current_messages = state['messages'] # This contains HumanMessage from start
            
            prompt_template = ChatPromptTemplate.from_messages([
                MessagesPlaceholder(variable_name="messages"), # This placeholder is crucial for LangGraph's message history
                ("system", f"You are a helpful customer service AI. The user's query has been analyzed, and their segment is '{customer_segment}'. A preliminary suggestion is: '{suggestion}'. Based on this, and the conversation history, formulate a friendly and helpful response. If the preliminary suggestion is sufficient, you can incorporate it directly. Otherwise, elaborate or rephrase to be most helpful to the user."),
            ])

            chain = prompt_template | self.llm | StrOutputParser()
            
            response = chain.invoke({"messages": current_messages}) # Pass the current list of messages to the prompt

            # Append AI's response to the messages in the state for the next turn
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
        # Ensure all keys of GraphState are provided.
        initial_state = GraphState(
            messages=chat_history_from_memory + [HumanMessage(content=user_input)],
            user_query=user_input,
            customer_segment="",
            suggestion="",
            error="",
            chat_history_str="" # Explicitly initialize
        )

        # Run the graph using invoke() to get the final state directly
        try:
            # invoke() returns the final state of the graph
            final_state = self.app.invoke(initial_state)
            print(f"Final state from invoke: {final_state}") # Debugging

            ai_response = "An error occurred during response generation." # Default fallback

            # Access messages from the final_state
            final_messages = final_state.get('messages', [])
            if final_messages:
                last_message = final_messages[-1]
                if isinstance(last_message, AIMessage):
                    ai_response = last_message.content
                else:
                    ai_response = "I processed your request, but couldn't formulate a complete AI response (last message not AIMessage)."
                    print(f"Warning: Last message in state was not an AIMessage: {last_message}")
            else:
                ai_response = "I processed your request, but no messages were generated by the AI."
                print("Warning: final_state['messages'] was empty or missing.")

            # Save the interaction to Langchain memory
            self.memory.save_context(
                {"input": user_input},
                {"output": ai_response}
            )
            return ai_response
        except Exception as e:
            print(f"Error during LangGraph invoke: {e}")
            # If an error occurs during invoke, return an error message
            return f"An error occurred while processing your request: {e}"