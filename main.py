import logging
import time
import schedule
import sys
import argparse # Import argparse
import json # Add json import
from datetime import date # Add datetime import

# Setup logging first
from logger_setup import setup_logging
setup_logging() 

# Import other modules after logging is configured
from connections_fetcher import fetch_latest_puzzle
from game_logic import Game
from ai_player import get_ai_player, SUPPORTED_MODELS # Import the map
from bluesky_poster import post_results_to_bluesky

logger = logging.getLogger(__name__)

# Use the keys from the SUPPORTED_MODELS map in ai_player for validation
# Filter out prefixed versions for cleaner CLI help text
ALLOWED_MODELS_FOR_CLI = sorted(SUPPORTED_MODELS.keys())

# Default model remains Gemini for now
DEFAULT_MODEL = "gemini-2.5-pro-exp-03-25"

def run_daily_puzzle(model_name: str, post_to_bluesky: bool):
    """Fetches, plays, and posts the result for the daily Connections puzzle using the specified model."""
    logger.info(f"--- Starting Daily Connections AI Run (Model: {model_name}, Post: {post_to_bluesky}) ---")

    # 1. Fetch Puzzle Data
    puzzle_data = fetch_latest_puzzle()
    if not puzzle_data:
        logger.info("No new puzzle data found or fetch failed. Ending run.")
        return

    puzzle_id = puzzle_data.get('id', 'Unknown')
    logger.info(f"Processing Puzzle ID: {puzzle_id}")

    # 2. Initialize Game and AI (Use factory function)
    try:
        game = Game(puzzle_data)
        # Use the factory function to get the correct player instance
        ai_player = get_ai_player(model_name=model_name) 
    except ValueError as e:
        logger.error(f"Failed to initialize game or AI for Puzzle ID {puzzle_id} (Model: {model_name}): {e}")
        return
    except Exception as e:
        logger.error(f"Unexpected error initializing game/AI for Puzzle ID {puzzle_id}: {e}", exc_info=True)
        return

    # 3. Play Loop (uses ai_player instance)
    logger.info("Starting AI play loop...")
    while not game.is_solved() and not game.is_lost():
        current_words = game.get_remaining_words()
        logger.info(f"Attempt {game.attempts + 1}/{game.MAX_ATTEMPTS}. Remaining words: {len(current_words)}")

        # Get AI guess from the instantiated player
        ai_guess = ai_player.get_ai_guess(current_words, game.log) 

        if not ai_guess:
            logger.error("AI failed to provide a valid guess. Stopping game.")
            # Mark game as lost? Or just log and break?
            # For now, break and the summary will reflect the state.
            break

        # Submit guess to game logic
        try:
            result = game.submit_guess(ai_guess)
            logger.info(f"AI ({model_name}) guessed {ai_guess}. Result: {result[0]}")

            # Optional delay between attempts
            # time.sleep(1)

        except Exception as e:
            logger.error(f"Error submitting guess {ai_guess}: {e}", exc_info=True)
            break # Stop if submitting causes an error

    # 4. Game Finished - Log Summary and Post
    logger.info("--- Game Finished ---")
    game_summary = game.get_summary()
    
    log_summary_str = f"Puzzle ID: {puzzle_id} | Solved: {game_summary['solved']} | Attempts: {game_summary['attempts_made']}"
    logger.info(log_summary_str)
    # Log the found groups for detail
    for group in game_summary.get('found_groups_details', []):
        logger.info(f"  - Found ({group['color']}): {group['category']} - {group['members']}")

    # Get AI category guesses if puzzle was solved
    category_guesses = {}
    if game_summary['solved']:
        try:
            category_guesses = ai_player.get_category_guesses(game_summary['found_groups_details'])
            logger.info("AI Category Guesses:")
            for color, guess in category_guesses.items():
                logger.info(f"  - {color}: {guess}")
        except Exception as e:
            logger.error(f"Error getting AI category guesses: {e}")

    # 5. Post to Bluesky (Conditionally)
    if post_to_bluesky:
        try:
            post_results_to_bluesky(game_summary, puzzle_id, model_name=model_name, category_guesses=category_guesses)
        except Exception as e:
            logger.error(f"Error during Bluesky posting: {e}", exc_info=True)
    else:
        logger.info("Skipping Bluesky post due to --no-post flag.")

    # 6. Save results to JSON
    results_file = 'results.json'
    try:
        # Prepare the new result entry
        new_result = {
            'date': date.today().isoformat(),
            'puzzleid': puzzle_id,
            'model': model_name,
            'success': game_summary['solved'],
            'attempts': game_summary['attempts_made'],
            # Extract the guesses from the game log
            'guesses': [attempt[1] for attempt in game.log if attempt[0] > 0], # attempt[1] is the words list, skip initial state
            'category_guesses': category_guesses if category_guesses else None
        }

        # Read existing data
        try:
            with open(results_file, 'r') as f:
                results_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # If file doesn't exist or is empty/invalid, start with an empty list
            results_data = []

        # Append new result
        results_data.append(new_result)

        # Write updated data back to file
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2) # Use indent for readability
        logger.info(f"Successfully saved results to {results_file}")

    except Exception as e:
        logger.error(f"Error saving results to {results_file}: {e}", exc_info=True)

    logger.info("--- Daily Connections AI Run Finished ---")


# --- Main Execution & Scheduling --- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the NYT Connections AI Player - an AI system that attempts to solve the daily NYT Connections puzzle.",
        epilog="Example: python main.py --model gemini-2.5-pro-exp-03-25 --run-once --no-post"
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default=DEFAULT_MODEL, 
        choices=ALLOWED_MODELS_FOR_CLI, 
        metavar='MODEL_NAME',
        help=f"The AI model to use for solving the puzzle. Default: {DEFAULT_MODEL}"
    )
    parser.add_argument(
        '--run-once', 
        action='store_true', 
        help="Run the puzzle solver once immediately and exit. If not specified, runs on a daily schedule at 07:00 local time."
    )
    parser.add_argument(
        '--post',
        dest='post_to_bluesky',
        action='store_true',
        default=True,
        help="Post results to Bluesky social network (default behavior)."
    )
    parser.add_argument(
        '--no-post',
        dest='post_to_bluesky',
        action='store_false',
        help="Do not post results to Bluesky social network."
    )
    
    args = parser.parse_args()
    selected_model = args.model
    run_once = args.run_once
    should_post = args.post_to_bluesky # Get the value from argparse

    # Allow running once via command line argument
    # run_once = len(sys.argv) > 1 and sys.argv[1] == '--run-once'

    if run_once:
        logger.info(f"Running the puzzle check once (Model: {selected_model}, Post: {should_post}).")
        run_daily_puzzle(model_name=selected_model, post_to_bluesky=should_post)
    else:
        # Schedule the job daily
        schedule_time = "07:00"
        logger.info(f"Scheduling daily puzzle run for {schedule_time} local time (Model: {selected_model}, Post: {should_post}).")
        # Use lambda to pass the model name and post flag to the scheduled job
        schedule.every().day.at(schedule_time).do(
            lambda: run_daily_puzzle(model_name=selected_model, post_to_bluesky=should_post)
        )

        # Run once immediately on start, then schedule future runs
        logger.info("Running initial check on startup...")
        run_daily_puzzle(model_name=selected_model, post_to_bluesky=should_post)

        logger.info("Entering scheduling loop. Press Ctrl+C to exit.")
        while True:
            schedule.run_pending()
            time.sleep(60) # Check every minute 