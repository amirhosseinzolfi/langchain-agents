import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.text import Text
from rich.theme import Theme

# --- Configuration ---
LOG_LEVEL = "INFO"  # Default log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
CONSOLE_WIDTH = 120 # Width of the console for formatting

# --- Initialize Rich Console ---
# Custom theme for different log types or components
custom_theme = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "error": "bold red",
    "debug": "dim green",
    "critical": "bold white on red",
    "agent_action": "bold blue",
    "tool_input": "italic yellow",
    "tool_output": "italic green",
    "db_operation": "purple",
    "user_input": "bold turquoise4"
})
console = Console(width=CONSOLE_WIDTH, theme=custom_theme)

# --- Setup Logger ---
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(message)s", # Basic format, RichHandler will do the styling
    datefmt="[%X]",       # Time format
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)] # show_path=False to keep it minimal
)

logger = logging.getLogger("rich")

# --- Custom Logging Functions (Optional but useful for specific styling) ---

def log_agent_action(message: str):
    logger.info(Text.from_markup(f"[agent_action]🤖 Agent Action: {message}[/agent_action]"))

def log_tool_input(tool_name: str, args):
    logger.info(Text.from_markup(f"[tool_input]🛠️ Using Tool: {tool_name} | Input: {args}[/tool_input]"))

def log_tool_output(tool_name: str, output: str):
    # Truncate long outputs for readability
    max_len = 200 
    display_output = output[:max_len] + "..." if len(output) > max_len else output
    logger.info(Text.from_markup(f"[tool_output]✨ Tool Result: {tool_name} | Output: {display_output}[/tool_output]"))

def log_db_operation(message: str):
    logger.info(Text.from_markup(f"[db_operation]📦 DB: {message}[/db_operation]"))

def log_user_prompt(message: str):
    # Using console.print for direct styling without log prefix if desired
    console.print(Text.from_markup(f"[user_input]👤 You: {message}[/user_input]"))

def log_agent_response(message: str):
    console.print(Text.from_markup(f"🤖 Agent: {message}"))
    
def log_error(message: str):
    logger.error(message)

def log_info(message: str):
    logger.info(message)

def log_warning(message: str):
    logger.warning(message)
    
def log_debug(message: str): # Add a debug log function
    logger.debug(message)

if __name__ == '__main__':
    # Example Usage
    log_info("This is an informational message.")
    log_warning("This is a warning message.")
    log_error("This is an error message.")
    log_debug("This is a debug message. Set LOG_LEVEL to DEBUG to see it.")
    
    console.print("--- Custom Loggers ---")
    log_agent_action("Thinking about what to do next.")
    log_tool_input("SaveInfoTool", {"information": "Buy milk", "category": "todo"})
    log_tool_output("SaveInfoTool", "Information 'Buy milk...' (ID: abc123) saved.")
    log_db_operation("Saved 1 new entry to chat_database.json")
    log_user_prompt("Hello there!")
    log_agent_response("Hi! How can I help you today?")
