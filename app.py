from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
import os
import uuid
from datetime import datetime

from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException # Optional, for more specific typing
from langchain import hub # Added for hub.pull

# Import database functions
import json_db_handler as db
import custom_logger as cl # Import custom logger
import chainlit as cl_chainlit # Import chainlit

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
Be polite and helpful. Always confirm actions taken.

**Output Format Instructions:**
When you need to use a tool or provide a final answer, you MUST follow this format:

Thought: [Your reasoning process and plan. Explain what you are trying to do and why.]
Action: [The name of the tool to use, e.g., SaveInfoTool, ListInfoTool, RemoveInfoTool. If you are providing a final answer to the user, use "Final Answer".]
Action Input: [If using a tool, the input to the tool as a JSON dictionary. For "Final Answer", this is the direct response to the user.]

Example of using a tool:
Thought: The user wants to save a to-do item. I need to extract the task description and assign a category.
Action: SaveInfoTool
Action Input: {"information": "Buy groceries for the week", "category": "todo"}

Example of providing a final answer:
Thought: The user asked a general question and I have the answer.
Action: Final Answer
Action Input: Hello! How can I assist you today?

If you make a mistake or an observation shows an error, think about what went wrong and try again, following the format.
If a tool is not needed, or you are responding to a greeting, use "Final Answer".
"""

# Comment out or remove the old prompt_template creation:
# prompt_template = ChatPromptTemplate.from_messages([...])

# Pull the base ReAct prompt
# This prompt already includes placeholders for 'tools', 'tool_names', 
# 'input', 'agent_scratchpad', and typically a system message.
base_react_prompt = hub.pull("hwchase17/react")

# Our SYSTEM_PROMPT_TEXT is already defined globally.
# We need to ensure our system message is used.
# Hub prompts are often ChatPromptTemplate objects. We can inspect their messages.
# Option A: If the hub prompt's first message is a SystemMessage, modify it.
# Option B: If more complex, reconstruct or prepend. Let's try to modify if possible.

# Let's inspect the messages and try to replace the system message part.
# This assumes the hub prompt's structure, which is common for 'hwchase17/react'.
# The 'hwchase17/react' prompt has its main instructions within a SystemMessage,
# but also has human/ai message placeholders. The core ReAct instructions are usually part of the system message.
# We want to insert our detailed system instructions (task, tool usage, ReAct format) there.

# A robust way is to rebuild the messages list from the hub prompt,
# replacing or augmenting its original system message content.
# The 'hwchase17/react' prompt has a structure like:
# [SystemMessage(...), MessagesPlaceholder(variable_name='chat_history', optional=True), HumanMessagePromptTemplate(...), MessagesPlaceholder(variable_name='agent_scratchpad')]
# The crucial part is that 'tools' and 'tool_names' are handled by the agent when it uses this prompt.

# Let's modify the system message content within the pulled prompt.
# The base_react_prompt.messages[0] is usually the SystemMessage.
if base_react_prompt.messages and isinstance(base_react_prompt.messages[0], SystemMessage):
    # Prepend our detailed system instructions to the existing content of the hub's system message.
    # This way, we keep any core ReAct formatting from the hub prompt and add our specifics.
    # Alternatively, replace it if SYSTEM_PROMPT_TEXT is comprehensive enough.
    # For 'hwchase17/react', its system message already contains core ReAct instructions.
    # It's often better to use OUR system message as the primary one if it defines the agent's persona and task.
    # Let's try replacing the content of the first system message.
    
    # Our SYSTEM_PROMPT_TEXT already includes detailed ReAct formatting instructions.
    # So, we can replace the hub prompt's system message content with ours.
    
    # Create a new list of messages to avoid modifying the hub object directly if it's cached.
    new_messages = list(base_react_prompt.messages)
    new_messages[0] = SystemMessage(content=SYSTEM_PROMPT_TEXT) # Replace system message
    
    prompt_template = ChatPromptTemplate.from_messages(new_messages)
    cl.log_info("Using Langchain Hub prompt 'hwchase17/react' and replaced its system message with our custom SYSTEM_PROMPT_TEXT.")

else:
    cl.log_warning("Could not find SystemMessage in the pulled hub prompt to replace. Using SYSTEM_PROMPT_TEXT with a basic structure. This might not be optimal for ReAct.")
    # Fallback if the hub prompt structure is unexpected
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=SYSTEM_PROMPT_TEXT), # Our custom system message
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    # IMPORTANT: This fallback might still raise the same ValueError if 'tools' and 'tool_names'
    # are not implicitly handled by create_react_agent when the prompt doesn't come from the hub
    # with the special partial_variables for tools. The hub.pull() method usually sets this up.
    # The primary approach (modifying hub prompt) is preferred.

# --- Create Agent ---
# The global agent and agent_executor are not used by Chainlit sessions anymore.
# They are defined per-session in on_chat_start.
# agent = create_react_agent(llm, tools, prompt_template)
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
# cl.log_info("Global agent initialized (will be overridden by session agent in Chainlit).")
cl.log_info("Global prompt_template is now configured using Langchain Hub 'hwchase17/react' with custom system message.")

# --- Main Interaction Loop ---
# This section has been removed to integrate with Chainlit.
# Chainlit will handle the user interaction and message handling.

# --- Chainlit Event Handlers ---

# --- Custom Chainlit Callback Handler for Agent Steps ---

# --- Custom Error Handling Function for ReAct Agent ---
def handle_react_parsing_error(error: Exception) -> ToolMessage:
    '''
    Handles parsing errors from the ReAct agent's LLM output.
    Formats the error as a ToolMessage to be fed back into the agent's scratchpad.
    '''
    # Extract the problematic LLM output if possible (often in error.llm_output or error.observation)
    # For now, we'll use a generic message based on the error string.
    # In more advanced scenarios, you might try to get `error.llm_output`
    error_content = f"Error parsing LLM output: {str(error)}. Please try to reformulate your thought and action."
    
    # It's important that this message is presented as an "observation" 
    # that the agent can then use in its next reasoning step.
    # Using a ToolMessage is conventional for observations.
    # The 'tool_call_id' isn't strictly necessary here unless your specific agent/tool setup requires it.
    # If the error object has an 'observation' field, that might be more appropriate.
    # Some parsing errors might put the problematic output in `error.observation`.
    observation = getattr(error, 'observation', error_content)
    
    cl.log_warning(f"ReAct Parsing Error. Feeding back to agent: {observation}")
    # Ensure the content is a string
    return ToolMessage(content=str(observation), name="ParsingErrorObservation") # Using a descriptive name for the pseudo-tool

class ChainlitCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self._current_step = None # To hold the current cl_chainlit.Step

    async def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        '''Called when the agent is about to perform an action.'''
        # Create a new step for this action
        self._current_step = cl_chainlit.Step(name=action.tool, type="tool")
        self._current_step.input = str(action.tool_input) # Log input to the step
        await self._current_step.send()
        
        # Log to server console via custom_logger
        cl.log_agent_action(f"Tool: {action.tool}, Input: {action.tool_input}")

    async def on_tool_end(self, output: str, name: str, **kwargs) -> None:
        '''Called when a tool has finished running.'''
        if self._current_step:
            self._current_step.output = str(output)
            await self._current_step.update() # Update the step with the output
            self._current_step = None # Reset for the next action
        
        # Log to server console via custom_logger
        cl.log_tool_output(name, str(output))

    async def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        '''Called when the agent has finished its thought process.'''
        # You could potentially add a final step here for the agent's thought process
        # if finish.log contains useful summary information.
        # For now, we'll just log it to the console.
        cl.log_debug(f"Agent finished with: {finish.log}")
        if self._current_step: # If there's an open step, close it.
             await self._current_step.update() # Or remove if it's empty
             self._current_step = None

    # Optional: You can also implement other methods like on_llm_start, on_llm_end, 
    # on_chain_start, on_chain_end, on_tool_error, etc., to provide more granular feedback.

@cl_chainlit.on_chat_start
async def start_chat():
    # LLM, Embeddings, global 'database', tool functions, 'tools' list, 
    # 'SYSTEM_PROMPT_TEXT', and 'prompt_template' are already initialized/defined globally.
    
    cl.log_info("New chat session started. Initializing agent executor with custom parsing error handler.")
    
    # Instantiate the callback handler
    chainlit_callback_handler = ChainlitCallbackHandler() # Assuming this is already defined
    
    session_agent = create_react_agent(llm, tools, prompt_template)
    session_agent_executor = AgentExecutor(
        agent=session_agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=handle_react_parsing_error, # Use the custom handler
        callbacks=[chainlit_callback_handler]
    )
    
    # Store the session-specific agent_executor and chat_history in the user's session.
    cl_chainlit.user_session.set("agent_executor", session_agent_executor)
    cl_chainlit.user_session.set("chat_history", []) # Initialize empty chat history for the session
    
    # Send a welcome message to the Chainlit UI.
    await cl_chainlit.Message(
        content="Hello! I am your Langchain JSON DB assistant. How can I help you today?",
        author="Agent" # You can customize the author name.
    ).send()
    cl.log_info("Welcome message sent to user in Chainlit UI.")


@cl_chainlit.on_message
async def on_message(message: cl_chainlit.Message):
    # Retrieve the agent_executor and chat_history from the user's session.
    agent_executor = cl_chainlit.user_session.get("agent_executor")
    chat_history = cl_chainlit.user_session.get("chat_history")

    if not agent_executor:
        cl.log_error("Agent executor not found in user session.")
        await cl_chainlit.Message(
            content="Sorry, there was an error initializing the agent. Please try refreshing the page.",
            author="Error"
        ).send()
        return

    cl.log_user_prompt(f"[Chainlit] User: {message.content}") # Log user input via custom logger

    # Construct the input for the agent
    agent_input = {
        "input": message.content,
        "chat_history": chat_history
    }

    # Send a thinking indicator to the UI
    thinking_message = cl_chainlit.Message(content="", author="Agent") # Empty message to update later
    await thinking_message.send()


    try:
        # Run the agent executor
        # Note: AgentExecutor.invoke is synchronous. To avoid blocking,
        # run it in a separate thread if it's long-running.
        # For many LLM calls, this might be acceptable, but for production, consider cl_chainlit.make_async
        
        # Using cl_chainlit.make_async to run the synchronous agent call in a separate thread
        response = await cl_chainlit.make_async(agent_executor.invoke)(agent_input)
        agent_output = response.get('output', "Sorry, I had trouble processing that.")

    except Exception as e:
        cl.log_error(f"Error during agent execution: {e}")
        agent_output = f"An error occurred: {e}"
        # Optionally, send a more user-friendly error message
        # await cl_chainlit.Message(content=f"Sorry, an error occurred: {e}", author="Error").send()
        # return # Or allow the generic response below


    # Update thinking message with actual response
    thinking_message.content = agent_output
    thinking_message.author = "Agent" # Ensure author is set if it changed
    await thinking_message.update() # Update the message in the UI

    cl.log_agent_response(f"[Chainlit] Agent: {agent_output}") # Log agent response

    # Update chat history
    chat_history.append(HumanMessage(content=message.content))
    chat_history.append(AIMessage(content=agent_output))
    
    # Keep chat_history relatively short if needed (optional, same as before)
    if len(chat_history) > 10: 
        chat_history = chat_history[-10:]
    cl_chainlit.user_session.set("chat_history", chat_history)
