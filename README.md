# Langchain JSON DB Agent with Chainlit UI

This project implements a conversational AI agent using Langchain that can interact with a local JSON database. The user interacts with the agent through a web UI powered by Chainlit.

## Features

- **Conversational Interface**: Chat with the agent in natural language via a Chainlit web UI.
- **JSON Database Interaction**:
    - **Save Information**: Tell the agent to remember pieces of information (e.g., tasks, notes, reminders). The agent extracts details and saves them to a `chat_database.json` file.
    - **List Information**: Ask the agent to list all stored information.
    - **Remove Information**: Instruct the agent to remove specific pieces of information by their ID.
- **Structured Data**: Information is stored with an ID, category, the information itself, and a timestamp.
- **Enhanced Logging**:
    - **Server-Side**: Detailed logs, including agent thoughts and actions, are printed to the server console using `rich` for enhanced readability.
    - **Chainlit UI**: Agent actions (like tool usage) are displayed as steps within the chat interface for better user visibility.
- **Powered by Langchain & Ollama**: Uses Langchain for the agent framework and OpenAI (configurable) for LLM capabilities. Ollama embeddings are configured but not actively used in the current agent setup beyond initialization.

## Setup

1.  **Clone the repository (if applicable).**

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    Make sure you have Python 3.8+ installed.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure LLM:**
    The application is configured to use an OpenAI-compatible API (currently pointing to `http://141.98.210.149:15203/v1` with model `gpt-4o` and API key `324`). This is set in `app.py`.
    - If you are using a different OpenAI-compatible endpoint or model, update the `ChatOpenAI` initialization in `app.py`.
    - The `OllamaEmbeddings` are initialized with `nomic-embed-text` but are not central to the current agent's DB operations. Ensure your Ollama service is running if you plan to extend its use.

## Running the Application

1.  **Ensure your virtual environment is activated.**

2.  **Run the Chainlit application:**
    ```bash
    chainlit run app.py -w
    ```
    - The `-w` flag enables auto-reloading when code changes are saved.
    - Chainlit will typically open the application in your web browser automatically (usually at `http://localhost:8000`).

3.  **Interact with the Chatbot:**
    - Open the URL provided by Chainlit in your browser.
    - Start chatting with the agent!

## Project Structure

- `app.py`: Contains the main Chainlit application logic, Langchain agent setup, and tool definitions.
- `json_db_handler.py`: Manages CRUD operations for the `chat_database.json` file.
- `custom_logger.py`: Implements the `rich`-based custom logging for server-side console output.
- `requirements.txt`: Lists all Python dependencies.
- `chat_database.json`: The JSON file where data is stored (will be created automatically on first save).
- `README.md`: This file.
