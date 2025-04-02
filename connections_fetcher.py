import requests
import logging
from datetime import date

# Use the configured logger
logger = logging.getLogger(__name__)

# URL for the Connections answers
# Using the main branch ref for potentially more up-to-date data
CONNECTIONS_URL = "https://raw.githubusercontent.com/Eyefyre/NYT-Connections-Answers/main/connections.json"

# Store the ID of the last processed puzzle to avoid re-playing
# In a real application, this might be stored in a file or database
_last_processed_puzzle_id = None

def fetch_latest_puzzle():
    """Fetches the latest Connections puzzle data.

    Returns:
        dict: The puzzle data (words and solutions) for the latest puzzle,
              or None if the request fails or no new puzzle is found.
    """
    global _last_processed_puzzle_id
    logger.info(f"Fetching Connections data from {CONNECTIONS_URL}")
    try:
        response = requests.get(CONNECTIONS_URL)
        response.raise_for_status()  # Raise an exception for bad status codes
        all_puzzles_data = response.json()

        if not all_puzzles_data:
            logger.warning("Fetched data is empty.")
            return None

        # Assuming the JSON is a list of puzzles, sorted newest first
        # Correction: Data is sorted OLDEST first, take the LAST element.
        # And each puzzle has an 'id'
        latest_puzzle = all_puzzles_data[-1] # Get the last element
        puzzle_id = latest_puzzle.get('id')
        puzzle_date_str = latest_puzzle.get('date') # Use 'date' key instead of 'print_date'

        # Check if the puzzle date matches today's date
        today_str = date.today().isoformat()
        if puzzle_date_str != today_str:
            logger.info(f"Latest puzzle in data is for {puzzle_date_str}, not today ({today_str}). Skipping.")
            # Optional: Could check the next few entries if needed, but the source
            #           usually updates promptly for the current day.
            return None

        # Check if we've already processed this puzzle ID
        # Note: This simple in-memory check resets if the script restarts.
        # A persistent store would be needed for true idempotency across runs.
        if puzzle_id is not None and puzzle_id == _last_processed_puzzle_id:
            logger.info(f"Puzzle ID {puzzle_id} has already been processed. Skipping.")
            return None

        logger.info(f"Successfully fetched new puzzle data for ID: {puzzle_id} ({puzzle_date_str})")
        _last_processed_puzzle_id = puzzle_id
        return latest_puzzle

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching Connections data: {e}")
        return None
    except ValueError as e: # Catches JSON decoding errors
        logger.error(f"Error decoding JSON data: {e}")
        return None
    except IndexError:
         logger.error("Fetched data appears to be an empty list.")
         return None

# Example Usage (for testing)
if __name__ == "__main__":
    from logger_setup import setup_logging
    setup_logging() # Setup logging for standalone testing
    puzzle = fetch_latest_puzzle()
    if puzzle:
        print("Fetched Puzzle Data:")
        # print(puzzle)
        print(f"ID: {puzzle.get('id')}")
        print(f"Date: {puzzle.get('date')}")
        print("Words:", puzzle.get('startingGroups')) # Structure might vary, adjust as needed
        print("Solutions:", puzzle.get('answers')) # Structure might vary
    else:
        print("Failed to fetch puzzle data or no new puzzle today.") 