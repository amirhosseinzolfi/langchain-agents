import json
import os
import custom_logger as cl # Import custom logger

DB_FILEPATH = 'chat_database.json'

def load_db(filepath=DB_FILEPATH):
    """Loads the JSON database from a file. If the file doesn't exist, creates an empty list."""
    cl.log_debug(f"Loading DB from {filepath}. Found: {os.path.exists(filepath)}")
    if not os.path.exists(filepath):
        cl.log_debug(f"File {filepath} not found, returning empty list.")
        return []
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            if not isinstance(data, list):
                cl.log_warning(f"Data in {filepath} is not a list, returning empty list.")
                return []
            cl.log_debug(f"Successfully loaded {len(data)} entries from {filepath}.")
            return data
    except (json.JSONDecodeError, IOError) as e:
        cl.log_error(f"Error loading DB from {filepath}: {e}. Returning empty list.")
        return []

def save_db(data, filepath=DB_FILEPATH):
    """Saves the current state of the database (list of entries) to the JSON file."""
    cl.log_debug(f"Saving DB to {filepath}. {len(data)} entries.")
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        cl.log_debug(f"Successfully saved DB to {filepath}.")
    except IOError as e:
        cl.log_error(f"Error saving DB to {filepath}: {e}")

def add_entry(db_data, new_entry):
    """Adds a new structured entry to the database (list of entries).
    Each entry is expected to be a dictionary.
    Returns the updated database data.
    """
    if not isinstance(db_data, list): # Initialize if not a list
        db_data = []
    db_data.append(new_entry)
    return db_data

def list_entries(db_data):
    """Returns a formatted string of all entries.
    If no entries, returns a specific message.
    """
    if not db_data:
        return "No entries found in the database."
    
    formatted_list = []
    for idx, entry in enumerate(db_data):
        formatted_entry = f"Entry {idx + 1}:\n"
        for key, value in entry.items():
            formatted_entry += f"  {key}: {value}\n"
        formatted_list.append(formatted_entry.strip())
    return "\n---\n".join(formatted_list)

def find_entry_by_id(db_data, entry_id):
    """Finds an entry by its 'id' field. Returns the entry or None.
    Assumes each entry has a unique 'id'.
    """
    if not isinstance(db_data, list):
        return None
    for entry in db_data:
        if isinstance(entry, dict) and entry.get('id') == entry_id:
            return entry
    return None

def remove_entry_by_id(db_data, entry_id):
    """Removes an entry by its 'id' field.
    Returns the updated database data and a boolean indicating success.
    Assumes each entry has a unique 'id'.
    """
    if not isinstance(db_data, list):
        return db_data, False # Return original data and False if not a list

    original_length = len(db_data)
    # Filter out the entry with the given id
    updated_db_data = [entry for entry in db_data if not (isinstance(entry, dict) and entry.get('id') == entry_id)]
    
    removed = len(updated_db_data) < original_length
    return updated_db_data, removed

# Example usage (optional, for testing within the module)
if __name__ == '__main__':
    # Test the functions
    # Set logger level to DEBUG to see all messages for this test
    # Note: This will only affect the logger if custom_logger.LOG_LEVEL is also DEBUG or lower.
    # Or, you can temporarily set the level for the 'rich' logger directly in custom_logger.py for testing.
    cl.log_info("--- Testing json_db_handler.py ---")
    
    # To see debug logs from here, ensure custom_logger.LOG_LEVEL is "DEBUG"
    # or by setting logging.getLogger("rich").setLevel("DEBUG") in custom_logger.py
    
    # Clean up existing DB file for a fresh test if it exists
    if os.path.exists(DB_FILEPATH):
        cl.log_debug(f"Removing existing test DB file: {DB_FILEPATH}")
        os.remove(DB_FILEPATH)

    data = load_db()
    cl.log_info(f"Initial data (after potential cleanup): {data}")

    entry1 = {"id": "task1", "description": "Buy groceries", "status": "pending"}
    data = add_entry(data, entry1)
    entry2 = {"id": "task2", "description": "Read a book", "status": "in-progress"}
    data = add_entry(data, entry2)
    save_db(data)
    cl.log_info(f"After adding entries: {data}")

    cl.log_info("\nListing entries:")
    # list_entries returns a string, so we log it directly
    cl.log_info(list_entries(data))

    found_entry = find_entry_by_id(data, 'task1')
    cl.log_info(f"\nFinding entry task1: {found_entry}")
    
    data, removed = remove_entry_by_id(data, 'task1')
    if removed:
        cl.log_info("\nEntry task1 removed.")
    else:
        cl.log_warning("\nEntry task1 not found for removal.")
    save_db(data)
    cl.log_info(f"After removing task1: {data}")

    cl.log_info("\nListing entries again:")
    cl.log_info(list_entries(data))
    
    cl.log_info("--- End of json_db_handler.py test ---")
    # Clean up the test DB file after testing if desired
    # if os.path.exists(DB_FILEPATH):
    #     cl.log_debug(f"Cleaning up test DB file: {DB_FILEPATH}")
    #     os.remove(DB_FILEPATH)
