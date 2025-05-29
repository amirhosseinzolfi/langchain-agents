from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
import os
import uuid
from datetime import datetime

from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Import database functions
import json_db_handler as db
import custom_logger as cl # Import custom logger

# --- Initialize LLM & Embeddings ---
llm = ChatOpenAI(
    base_url="http://141.98.210.149:15203/v1",
    model_name="gpt-4o",
    temperature=0.5,
    api_key="324"
)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
cl.log_info("LLM and Embeddings initialized.")

# --- Load Database ---
DB_FILEPATH = 'chat_database.json'
database = db.load_db(DB_FILEPATH) # This is now a global variable
cl.log_info(f"Database loaded. Contains {len(database)} entries.")

# --- Define Agent Tool Functions ---
def save_info_tool_func(information: str, category: str, item_id: str = None):
    '''Tool to save a new piece of information to the database.'''
    global database
    if not item_id or item_id.strip() == "": # Ensure item_id is not just whitespace
        item_id = uuid.uuid4().hex[:8]
    
    new_entry = {
        "id": item_id,
        "category": category,
        "info": information,
        "timestamp": datetime.now().isoformat()
    }
    database = db.add_entry(database, new_entry)
    db.save_db(database, DB_FILEPATH)
    cl.log_db_operation(f"Saved entry ID {item_id} to database.")
    return f"Information '{information[:30]}...' (ID: {item_id}) saved under category '{category}'."

def list_info_tool_func():
    '''Tool to list all information currently stored in the database.'''
    cl.log_db_operation("Listing entries from database.")
    return db.list_entries(database)

def remove_info_tool_func(item_id_to_remove: str):
    '''Tool to remove a piece of information from the database using its ID.'''
    global database
    updated_db, removed = db.remove_entry_by_id(database, item_id_to_remove)
    if removed:
        database = updated_db
        db.save_db(database, DB_FILEPATH)
        cl.log_db_operation(f"Removed entry ID {item_id_to_remove} from database.")
        return f"Information with ID '{item_id_to_remove}' removed successfully."
    else:
        cl.log_warning(f"Attempted to remove non-existent entry ID {item_id_to_remove}.")
        return f"Error: No information found with ID '{item_id_to_remove}', or database was initially empty/not a list."

tools = [
    Tool(
        name="SaveInfoTool",
        func=save_info_tool_func,
        description="Use this tool to save a new piece of information. Requires 'information' (the content) and 'category' (e.g., 'todo', 'note'). Optionally, you can provide an 'item_id'. If no 'item_id' is provided, one will be generated automatically."
    ),
    Tool(
        name="ListInfoTool",
        func=list_info_tool_func,
        description="Use this tool to list all pieces of information stored in the database."
    ),
    Tool(
        name="RemoveInfoTool",
        func=remove_info_tool_func,
        description="Use this tool to remove a piece of information by its 'item_id_to_remove'."
    )
]

# --- System Prompt ---
SYSTEM_PROMPT_TEXT = """You are a helpful assistant that manages a JSON database.
Your tasks are:
1. Chat with the user.
2. Extract specific information from user messages if it seems like a piece of data they want to remember (e.g., a to-do item, a note).
   When saving, you must provide the 'information' and a 'category' (e.g., 'todo', 'note', 'reminder'). If the user provides an ID, use it as 'item_id'; otherwise, an ID will be generated automatically by the SaveInfoTool.
3. Save this information to a JSON database using the 'SaveInfoTool'.
4. List information from the database using 'ListInfoTool' when the user asks.
5. Remove information from the database using 'RemoveInfoTool' when the user asks, using the 'item_id_to_remove'.

If the user asks to remove an item but doesn't provide an ID, ask them for the ID. You can list items to help them find the ID.
Be polite and helpful. Always confirm actions taken."""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT_TEXT),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# --- Create Agent ---
agent = create_react_agent(llm, tools, prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
cl.log_info("Agent initialized.")

# --- Main Interaction Loop ---
if __name__ == '__main__':
    cl.log_info("Starting chat with the Langchain JSON DB Agent...")
    chat_history = [] 

    while True:
        raw_user_input = input("👤 You: ") 
        if raw_user_input.lower() in ['exit', 'quit']:
            cl.log_info("Exiting chat.")
            break
        cl.log_user_prompt(raw_user_input) # Log it with styling
        
        try:
            response = agent_executor.invoke({
                "input": raw_user_input,
                "chat_history": chat_history 
            })
            agent_output = response.get('output', "Sorry, I had trouble processing that.")
        except Exception as e:
            # This will catch errors during agent execution, including tool errors if not caught by handle_parsing_errors
            agent_output = f"An error occurred: {e}"
            cl.log_error(f"Agent execution error: {e}") # For more detailed debugging if needed

        cl.log_agent_response(agent_output)
        
        chat_history.append(HumanMessage(content=raw_user_input))
        chat_history.append(AIMessage(content=agent_output))
        
        if len(chat_history) > 10: 
            chat_history = chat_history[-10:]
