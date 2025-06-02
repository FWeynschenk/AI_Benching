import logging
import time
from atproto import Client, models

from config import BLUESKY_HANDLE, BLUESKY_PASSWORD

logger = logging.getLogger(__name__)

# Constants
MAX_POST_LENGTH = 300 # Bluesky character limit
# AI_MODEL_NAME = "Gemini 1.5 Pro" # Removed - name is now passed
COLOR_EMOJIS = {
    "Yellow": "🟨",
    "Green": "🟩",
    "Blue": "🟦",
    "Purple": "🟪",
    "Unknown": "⬜️" # Fallback
}

# Mapping from technical model names to pretty names for posts
MODEL_PRETTY_NAMES = {
    # Gemini
    "gemini-1.5-pro": "Gemini 1.5 Pro",
    "gemini-1.5-flash": "Gemini 1.5 Flash",
    "gemini-1.5-flash-8b": "Gemini 1.5 Flash (8B)", 
    "gemini-2.0-flash": "Gemini 2.0 Flash", 
    "gemini-2.0-flash-lite": "Gemini 2.0 Flash Lite", 
    "gemini-2.5-pro-preview-05-06": "Gemini 2.5 Pro",
    "gemini-2.5-pro-exp-03-25": "Gemini 2.5 Pro Exp", 
    "gemini-2.5-flash-preview-05-20": "Gemini 2.5 Flash",
    # OpenAI (Curated list)
    "o4-mini": "OpenAI o4 Mini",
    "o3-mini": "OpenAI o3 Mini",
    "o1-mini": "OpenAI o1 Mini",
    "o1-pro": "OpenAI o1 Pro",
    "o1": "OpenAI o1",
    "o3": "OpenAI o3",
    "gpt-4o-mini": "OpenAI GPT-4o Mini",
    "gpt-4o": "OpenAI GPT-4o",
    "gpt-4.5-preview": "OpenAI GPT-4.5 Preview",
    # Anthropic
    "claude-3-7-sonnet-20250219": "Claude 3.7 Sonnet",
    "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet",
    "claude-3-5-haiku-20241022": "Claude 3.5 Haiku",
    "claude-3-opus-20240229": "Claude 3 Opus",
    "claude-sonnet-4-20250514": "Claude Sonnet 4",
}

def _create_bluesky_client() -> Client | None:
    """Creates and logs in the Bluesky client."""
    if not BLUESKY_HANDLE or not BLUESKY_PASSWORD:
        logger.error("Bluesky handle or password not configured. Cannot post results.")
        return None

    try:
        client = Client()
        logger.info(f"Attempting to log in to Bluesky as {BLUESKY_HANDLE}...")
        client.login(BLUESKY_HANDLE, BLUESKY_PASSWORD)
        logger.info("Bluesky login successful.")
        return client
    except Exception as e:
        logger.error(f"Failed to log in to Bluesky: {e}")
        return None

def _format_results_for_post(game_summary: dict, puzzle_id: str | int, model_name: str, category_guesses: dict[str, str] | None = None) -> str:
    """Formats the game summary into a concise post for Bluesky, including attempt grid."""
    status = "Solved" if game_summary['solved'] else "Did not solve"
    attempts = game_summary['attempts_made']
    max_attempts = game_summary['max_attempts']
    puzzle_identifier = f"Connections #{puzzle_id}" if puzzle_id else "Connections"
    word_map = game_summary.get('word_to_emoji_map', {})
    attempt_log = game_summary.get('attempt_log', [])

    # Get the pretty name, fallback to the technical name if not found
    pretty_model_name = MODEL_PRETTY_NAMES.get(model_name, model_name)

    lines = []
    lines.append(f"{puzzle_identifier} - {pretty_model_name} Result: {status} in {attempts} attempts.")

    # Build the attempt grid
    attempt_grid_lines = []
    for attempt_data in attempt_log:
        attempt_num, guessed_words, result_status, _, _ = attempt_data
        row = ""
        # Ensure guessed_words is a list/tuple
        if isinstance(guessed_words, (list, tuple)):
            for word in guessed_words:
                 # Lookup word (uppercase) in map, default to unknown if not found (shouldn't happen)
                 row += word_map.get(word.upper(), '⬜️')
            if len(row) == 4: # Ensure we got 4 emojis
                attempt_grid_lines.append(row)
            else:
                logger.warning(f"Could not generate full emoji row for attempt {attempt_num}: {guessed_words}")
                attempt_grid_lines.append("⬜️⬜️⬜️⬜️") # Placeholder for invalid attempt data
        else:
            logger.warning(f"Invalid guessed_words format in attempt log for attempt {attempt_num}: {guessed_words}")
            attempt_grid_lines.append("????") # Placeholder for bad data

    # Add category guesses next to the emoji rows if available
    if False and category_guesses and game_summary['solved']: # TODO wait for bsky to support some kind of spoiler tag, then remove False.
        # Find the first correct attempt for each color
        color_to_attempt = {}
        for attempt_data in attempt_log:
            _, _, status, category, color = attempt_data
            if status == "CORRECT" and color not in color_to_attempt:
                color_to_attempt[color] = attempt_data

        # Create a mapping of emoji row to category guess
        emoji_to_guess = {}
        for color, guess in category_guesses.items():
            if color in color_to_attempt:
                attempt_num, words, _, _, _ = color_to_attempt[color]
                row = ""
                for word in words:
                    row += word_map.get(word.upper(), '⬜️')
                emoji_to_guess[row] = guess

        # Add guesses to the grid lines
        for i, row in enumerate(attempt_grid_lines):
            if row in emoji_to_guess:
                attempt_grid_lines[i] = f"{row} - {emoji_to_guess[row]}"

    if attempt_grid_lines:
        # Join the grid lines with newlines
        attempt_grid_str = "\n".join(attempt_grid_lines)
        lines.append(attempt_grid_str)

    # Combine lines, ensuring it fits within the character limit
    full_post = "\n".join(lines)

    # Truncation logic (prioritize keeping the status line and as much grid as possible)
    if len(full_post) > MAX_POST_LENGTH:
        logger.warning(f"Generated post exceeds Bluesky character limit ({len(full_post)}/{MAX_POST_LENGTH}). Truncating.")
        status_line = lines[0]
        # Calculate remaining chars for grid + ellipsis
        available_chars_for_grid = MAX_POST_LENGTH - len(status_line) - 1 - 3 # -1 for newline, -3 for "..."
        
        if available_chars_for_grid > 0:
            truncated_grid = attempt_grid_str[:available_chars_for_grid] + "..."
            full_post = f"{status_line}\n{truncated_grid}"
        else: # If status line itself is too long (highly unlikely)
            full_post = status_line[:MAX_POST_LENGTH - 3] + "..."

    logger.info(f"Formatted Bluesky post (Model: {pretty_model_name}): \n{full_post}")
    return full_post

def post_results_to_bluesky(game_summary: dict, puzzle_id: str | int, model_name: str, category_guesses: dict[str, str] | None = None, max_retries=2):
    """Posts the formatted game results to Bluesky.

    Args:
        game_summary: The summary dictionary from the Game object.
        puzzle_id: The ID of the puzzle played.
        model_name: The name of the AI model used for this run.
        category_guesses: Dictionary mapping color to AI's guessed category name.
        max_retries: Number of times to retry posting on failure.
    """
    client = _create_bluesky_client()
    if not client:
        return # Error already logged

    post_text = _format_results_for_post(game_summary, puzzle_id, model_name, category_guesses)

    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting to post to Bluesky (Attempt {attempt + 1}/{max_retries})...")
            # Use client.send_post for more control if needed (e.g., adding images, links)
            client.post(text=post_text)
            logger.info("Successfully posted results to Bluesky.")
            return # Success

        except Exception as e:
            logger.error(f"Failed to post to Bluesky on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error("Max retries reached. Failed to post results to Bluesky.")

# Example Usage (for testing)
if __name__ == "__main__":
    from logger_setup import setup_logging
    setup_logging()
    # Import the list of models for testing
    from main import DEFAULT_MODEL, ALLOWED_MODELS

    # Dummy data needs word_to_emoji_map now for testing _format_results_for_post
    test_word_map = {
        "A1": "🟨", "A2": "🟨", "A3": "🟨", "A4": "🟨",
        "B1": "🟩", "B2": "🟩", "B3": "🟩", "B4": "🟩",
        "C1": "🟦", "C2": "🟦", "C3": "🟦", "C4": "🟦",
        "D1": "🟪", "D2": "🟪", "D3": "🟪", "D4": "🟪",
    }
    test_summary_solved = {
        "solved": True,
        "attempts_made": 5,
        "max_attempts": 20,
        "found_groups_details": [
            {"category": "Group A", "members": [], "color": "Yellow", "difficulty": 0},
            {"category": "Group B", "members": [], "color": "Green", "difficulty": 1},
            {"category": "Group C", "members": [], "color": "Blue", "difficulty": 2},
            {"category": "Group D", "members": [], "color": "Purple", "difficulty": 3}
        ],
        "attempt_log": [
            (1, ["D1", "D2", "C1", "C2"], "INCORRECT", None, None), # 🟪🟪🟦🟦
            (2, ["D1", "D2", "D3", "D4"], "CORRECT", "Group D", "Purple"), # 🟪🟪🟪🟪
            (3, ["C1", "C2", "C3", "C4"], "CORRECT", "Group C", "Blue"), # 🟦🟦🟦🟦
            (4, ["B1", "B2", "B3", "B4"], "CORRECT", "Group B", "Green"), # 🟩🟩🟩🟩
            (5, ["A1", "A2", "A3", "A4"], "CORRECT", "Group A", "Yellow") # 🟨🟨🟨🟨
        ],
        "word_to_emoji_map": test_word_map
    }

    test_summary_unsolved = {
        "solved": False,
        "attempts_made": 3,
        "max_attempts": 20,
        "found_groups_details": [
            {"category": "Group B", "members": [], "color": "Green", "difficulty": 1}
        ],
        "attempt_log": [
            (1, ["A1", "D1", "C1", "A2"], "INCORRECT", None, None), # 🟨🟪🟦🟨
            (2, ["B1", "B2", "B3", "B4"], "CORRECT", "Group B", "Green"), # 🟩🟩🟩🟩
            (3, ["D1", "D2", "A3", "C2"], "ONE_AWAY", None, None), # 🟪🟪🟨🟦
        ],
        "word_to_emoji_map": test_word_map
    }
    
    test_puzzle_id = 420

    # Test formatting with a few different models
    for test_model in [DEFAULT_MODEL, ALLOWED_MODELS[1], "gemini-2.5-pro-exp-03-25"]: # Test default, flash, and exp
        if test_model not in MODEL_PRETTY_NAMES:
             print(f"Skipping format test for {test_model} - missing pretty name mapping.")
             continue
        print(f"\n--- Testing Post Formatting (Model: {test_model}) ---")
        _format_results_for_post(test_summary_solved, test_puzzle_id, test_model)
        _format_results_for_post(test_summary_unsolved, test_puzzle_id, test_model)

    # print("\n--- Testing Actual Post (Requires .env configuration) ---")
    # uncomment below to test live posting
    # if BLUESKY_HANDLE and BLUESKY_PASSWORD:
    #      post_results_to_bluesky(test_summary_solved, test_puzzle_id, test_model_name)
    #      time.sleep(2) # Avoid rate limiting
    #      post_results_to_bluesky(test_summary_unsolved, test_puzzle_id, test_model_name)
    # else:
    #      print("Skipping actual post test: Bluesky credentials not found in .env") 
