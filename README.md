# PCI Query Flow Chatbot 

## Overview

This project extends the previous sentiment-aware chatbot by implementing the "Medium Level" task: a **PCI (Predictive Customer Intelligence) Query Flow** using **LangGraph**. This advanced workflow allows the chatbot to analyze customer queries, segment customers based on mock intelligence, and provide tailored product or offer suggestions, all while maintaining conversational context. The chatbot features a Streamlit-powered web interface for interactive use.

## Features

* **LangGraph Workflow**: Utilizes LangGraph to define a sophisticated, multi-node, stateful AI agent workflow, allowing for complex conversational flows and decision-making.
* **Conversation Memory**: Integrates Langchain's `ConversationBufferMemory` (or `ConversationSummaryMemory`) to ensure the chatbot remembers past interactions, facilitating coherent and continuous dialogues within the LangGraph flow.
* **Mock PCI Logic**: Includes a simulated "Predictive Customer Intelligence" module that performs rule-based customer segmentation (e.g., `churn_risk`, `price_sensitive`, `high_value_prospect`, `new_customer`, `standard`) and generates relevant product/offer suggestions.
* **Adaptive LLM Responses**: The core LLM within the LangGraph is prompted to integrate the detected customer segment and suggested actions into its natural language responses, providing personalized assistance.
* **Modular Design**: The codebase is organized into distinct Python modules for clarity, maintainability, and scalability.
* **Streamlit UI**: Provides an intuitive and user-friendly web interface for real-time interaction with the PCI chatbot.
* **Secure API Key Management**: Employs Streamlit's `st.secrets` mechanism for safely loading and using API keys, suitable for both local development and deployment environments.

## Modular Architecture

The project is structured into the following Python files and configuration:

* `app.py`: The main Streamlit application. It handles UI rendering, loads API keys from `st.secrets`, initializes the `LangGraphAgent`, and manages user input and chatbot output.
* `langgraph_agent.py`: Defines the core LangGraph agent. It sets up the workflow, defines individual nodes for query processing, customer segmentation, and suggestion generation, and manages the flow between them. It also integrates with Langchain's LLMs and memory.
* `pci_mock_logic.py`: Contains the mock business logic for Predictive Customer Intelligence, including functions for customer segmentation and generating tailored suggestions based on predefined rules.
* `state_definition.py`: Defines the `GraphState` (a `TypedDict`), which represents the internal state that persists and is passed between different nodes within the LangGraph workflow.
* `.streamlit/secrets.toml`: A configuration file used by Streamlit to securely store sensitive information like API keys.

## Setup

Follow these steps to set up and run the PCI Query Flow Chatbot locally.
### 1. Clone the repo and create the environment
### 2. Install Dependencies
```pip install -r requirements.txt ```

### 3. Create .streamlit/secrets.toml
`GOOGLE_API_KEY = "your_gemini_api_key_here"`

### 4. Run
`streamlit run app.py`

## Technical Details
- LangGraph for Stateful Workflows: The core of Task 2 is the StateGraph from LangGraph. It allows defining a directed graph of nodes (query_processing_node, customer_segmentation_node, suggestion_node) and edges that dictate the flow of execution based on a shared, mutable GraphState. This enables complex, multi-step reasoning and interaction.

- GraphState: A TypedDict that acts as the single source of truth for the agent's state. It carries information like conversation messages, user query, customer segment, and suggestions across different nodes.

- Mock PCI Logic: The pci_mock_logic.py module demonstrates how business logic (e.g., customer segmentation and next-best-action suggestions) can be integrated into the AI workflow without needing complex machine learning models for this specific task. It uses simple rule-based classifications.

- LLM Integration: Langchain's LLM (ChatGoogleGenerativeAI or ChatOpenAI) is used within the suggestion_node to formulate natural and coherent responses that incorporate the insights from the PCI logic and the conversation history.

- Separate Memory Management: While LangGraph manages its internal state, a separate ConversationBufferMemory instance is used to persist the long-term conversation history, which is loaded at the beginning of each run_agent call and saved at the end. This ensures the LLM has full context over time.