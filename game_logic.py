import logging
from collections import defaultdict
from bluesky_poster import COLOR_EMOJIS

logger = logging.getLogger(__name__)

class Game:
    """Manages the state and logic of a NYT Connections game."""

    MAX_ATTEMPTS = 8
    DIFFICULTY_COLORS = {0: "Yellow", 1: "Green", 2: "Blue", 3: "Purple"}

    def __init__(self, puzzle_data: dict):
        """Initializes the game with puzzle data.

        Args:
            puzzle_data: The dictionary containing puzzle details fetched from the source,
                         expected to have 'answers'.
        """
        if not puzzle_data or 'answers' not in puzzle_data:
            raise ValueError("Invalid puzzle data provided to Game constructor (missing 'answers').")

        self._process_puzzle_data(puzzle_data)

        self.remaining_words = set(self.all_words)
        self.found_groups = [] # Stores dicts of found solutions
        self.attempts = 0
        self.log = [] # Stores tuples of (attempt_number, guessed_words, result, category, color)

        logger.info(f"Game initialized for puzzle ID {puzzle_data.get('id', 'N/A')}. Words: {self.all_words}")


    def _process_puzzle_data(self, puzzle_data: dict):
        """Processes the raw puzzle data into internal game structures."""
        try:
            # Flatten startingGroups into a set of 16 words
            self.all_words = set()
            if not isinstance(puzzle_data['answers'], list) or len(puzzle_data['answers']) != 4:
                raise ValueError("Expected 'answers' to be a list of 4 groups.")

            for group_data in puzzle_data['answers']:
                members = group_data.get('members')
                if not members or not isinstance(members, list) or len(members) != 4:
                    raise ValueError(f"Invalid group data in 'answers': {group_data}")
                self.all_words.update(members)

            if len(self.all_words) != 16:
                 logger.warning(f"Expected 16 starting words after processing answers, but found {len(self.all_words)}. Puzzle ID: {puzzle_data.get('id')}")
                 # Consider if this should be a fatal error or just a warning
                 # raise ValueError(f"Expected 16 starting words, found {len(self.all_words)}")

            self.solutions = [] # List of {'category': str, 'members': set[str], 'difficulty': int}
            self.word_to_solution = {} # word -> {'category': str, 'difficulty': int}
            self.category_to_solution = {} # category -> {'members': set[str], 'difficulty': int}

            # Use the structure already parsed for self.all_words
            for i, group_data in enumerate(puzzle_data['answers']):
                category = group_data.get('group') # Use 'group' for category name
                members = group_data.get('members')
                # Assign difficulty based on 'level' key (0-3 = Yellow-Purple)
                difficulty = group_data.get('level', i) # Use 'level' for difficulty

                if not category or not members: # Already checked members list validity above
                    raise ValueError(f"Invalid group data in 'answers': {group_data}")

                member_set = set(members)
                solution_dict = {'category': category, 'members': member_set, 'difficulty': difficulty}
                self.solutions.append(solution_dict)
                self.category_to_solution[category] = {'members': member_set, 'difficulty': difficulty}

                for word in members:
                    # Redundant check as we built all_words from answers, but safe
                    # if word not in self.all_words:
                    #      raise ValueError(f"Word '{word}' in answers not found when building all_words.")
                    if word in self.word_to_solution:
                         raise ValueError(f"Word '{word}' appears in multiple solutions.")
                    self.word_to_solution[word] = {'category': category, 'difficulty': difficulty}

            if len(self.word_to_solution) != 16:
                 raise ValueError(f"Processed solutions map to {len(self.word_to_solution)} unique words, expected 16.")

        except KeyError as e:
            logger.error(f"Missing key in puzzle data: {e}")
            raise ValueError(f"Missing key in puzzle data: {e}") from e
        except Exception as e:
            logger.error(f"Error processing puzzle data: {e}")
            raise


    def get_remaining_words(self) -> list[str]:
        """Returns the list of words currently available to guess."""
        return sorted(list(self.remaining_words))

    def is_solved(self) -> bool:
        """Checks if all four groups have been found."""
        return len(self.found_groups) == 4

    def is_lost(self) -> bool:
        """Checks if the maximum number of attempts has been reached."""
        return self.attempts >= self.MAX_ATTEMPTS

    def submit_guess(self, guessed_words: list[str]) -> tuple:
        """Processes a player's guess.

        Args:
            guessed_words: A list of exactly four words.

        Returns:
            A tuple indicating the result:
            - ("INVALID_GUESS", message): If the guess is invalid.
            - ("ALREADY_FOUND", category, color): If the group was already found.
            - ("CORRECT", category, color): If the guess is a correct group.
            - ("ONE_AWAY",): If exactly three words belong to the same group.
            - ("INCORRECT",): If the guess is incorrect.
            - ("LOST",): If max attempts reached after this guess.
        """
        if self.is_solved():
            logger.warning("Attempt submitted after game was already solved.")
            return ("INVALID_GUESS", "Game already solved")
        if self.is_lost():
            logger.warning("Attempt submitted after game was already lost.")
            return ("INVALID_GUESS", "Maximum attempts reached")

        # --- Validation ---
        if not isinstance(guessed_words, list) or len(guessed_words) != 4:
            return ("INVALID_GUESS", "Guess must be a list of 4 words.")

        guessed_set = set(guessed_words)
        if len(guessed_set) != 4:
             return ("INVALID_GUESS", "Guess must contain 4 unique words.")

        # Check if all guessed words are currently available
        unknown_words = guessed_set - self.remaining_words
        if unknown_words:
            return ("INVALID_GUESS", f"Unknown or already used words: {unknown_words}")

        # --- Process Attempt ---
        self.attempts += 1
        result_category = None
        result_color = None
        result_status = "INCORRECT" # Default

        # Check if correct
        for solution in self.solutions:
            if guessed_set == solution['members']:
                # Check if this group was somehow already found (shouldn't happen with remaining_words logic)
                if solution in self.found_groups:
                     logger.warning(f"Guessed group '{solution['category']}' which was already found.")
                     result_status = "ALREADY_FOUND" # Treat as already found, don't penalize attempt technically but don't reveal again
                else:
                    self.found_groups.append(solution)
                    self.remaining_words -= guessed_set
                    result_status = "CORRECT"
                    result_category = solution['category']
                    result_color = self.DIFFICULTY_COLORS.get(solution['difficulty'], "Unknown")
                    logger.info(f"Attempt {self.attempts}: Correct! Found group '{result_category}' ({result_color}). Words: {guessed_words}")
                break # Found the match

        # If not correct, check for "one away"
        if result_status == "INCORRECT":
            group_counts = defaultdict(int)
            for word in guessed_set:
                # Find which solution group the word belongs to
                sol_category = self.word_to_solution[word]['category']
                group_counts[sol_category] += 1

            if max(group_counts.values()) == 3:
                result_status = "ONE_AWAY"
                logger.info(f"Attempt {self.attempts}: One away! Guessed: {guessed_words}")
            else:
                 logger.info(f"Attempt {self.attempts}: Incorrect. Guessed: {guessed_words}")


        # Log the attempt details
        self.log.append((self.attempts, guessed_words, result_status, result_category, result_color))

        # Check for game over conditions
        if self.is_solved():
            logger.info(f"Game solved in {self.attempts} attempts!")
            # Return the result of the final correct guess
            return (result_status, result_category, result_color)
        elif self.is_lost():
            logger.info(f"Game lost after {self.attempts} attempts.")
            return ("LOST",) # Special status for losing on the last attempt

        # Return the status of this specific attempt
        if result_status == "CORRECT":
            return ("CORRECT", result_category, result_color)
        elif result_status == "ONE_AWAY":
            return ("ONE_AWAY",)
        elif result_status == "ALREADY_FOUND":
             # Maybe return the details so the AI knows?
             sol = self.category_to_solution[list(group_counts.keys())[0]] # Get the solution details
             cat = sol['category']
             clr = self.DIFFICULTY_COLORS.get(sol['difficulty'], "Unknown")
             return ("ALREADY_FOUND", cat, clr)
        else: # INCORRECT
            return ("INCORRECT",)


    def get_summary(self) -> dict:
        """Generates a summary of the finished game."""
        # Create mapping from word to its solution color emoji
        word_to_emoji_map = {}
        for word, solution_info in self.word_to_solution.items():
            difficulty = solution_info['difficulty']
            color_name = self.DIFFICULTY_COLORS.get(difficulty, "Unknown")
            emoji = COLOR_EMOJIS.get(color_name, '⬜️') # Use constant from bluesky_poster?
            # For now, duplicate the emoji map here or assume standard emojis.
            # Let's define the emojis locally for now to avoid cross-dependency
            local_color_emojis = {"Yellow": "🟨", "Green": "🟩", "Blue": "🟦", "Purple": "🟪", "Unknown": "⬜️"}
            emoji = local_color_emojis.get(color_name, '⬜️')
            word_to_emoji_map[word.upper()] = emoji # Store with upper case keys for easier lookup

        summary = {
            "solved": self.is_solved(),
            "attempts_made": self.attempts,
            "max_attempts": self.MAX_ATTEMPTS,
            "found_groups_details": [],
            "attempt_log": self.log,
            "word_to_emoji_map": word_to_emoji_map # Add the new map
        }
        # Order found groups by difficulty
        sorted_found_groups = sorted(self.found_groups, key=lambda x: x['difficulty'])
        summary["found_groups_details"] = [
            {
                "category": g['category'],
                "members": sorted(list(g['members'])),
                "color": self.DIFFICULTY_COLORS.get(g['difficulty'], "Unknown"),
                "difficulty": g['difficulty']
            } for g in sorted_found_groups
        ]
        return summary

# Example Usage (for testing - requires a sample puzzle structure)
if __name__ == "__main__":
    # Dummy puzzle data for testing
    test_puzzle = {
        "id": 0,
        "date": "2024-01-01",
        "answers": [
            {"level": 0, "group": "Group A", "members": ["A1", "A2", "A3", "A4"]},
            {"level": 1, "group": "Group B", "members": ["B1", "B2", "B3", "B4"]},
            {"level": 2, "group": "Group C", "members": ["C1", "C2", "C3", "C4"]},
            {"level": 3, "group": "Group D", "members": ["D1", "D2", "D3", "D4"]}
        ]
    }

    from logger_setup import setup_logging
    setup_logging() # Setup logging for standalone testing

    game = Game(test_puzzle)
    print("Initial words:", game.get_remaining_words())

    # Test guesses
    print("Guess 1 (Correct Yellow):", game.submit_guess(["A1", "A2", "A3", "A4"]))
    print("Remaining words:", game.get_remaining_words())
    print("Guess 2 (Incorrect):", game.submit_guess(["B1", "B2", "C1", "C2"]))
    print("Guess 3 (One Away):", game.submit_guess(["B1", "B2", "B3", "C1"]))
    print("Guess 4 (Correct Green):", game.submit_guess(["B1", "B2", "B3", "B4"]))
    print("Remaining words:", game.get_remaining_words())
    print("Guess 5 (Invalid - too few):", game.submit_guess(["C1", "C2", "C3"]))
    print("Guess 6 (Invalid - unknown):", game.submit_guess(["C1", "C2", "C3", "X1"]))
    print("Guess 7 (Invalid - already used):", game.submit_guess(["A1", "C1", "C2", "C3"]))
    print("Guess 8 (Correct Blue):", game.submit_guess(["C1", "C2", "C3", "C4"]))
    print("Guess 9 (Correct Purple):", game.submit_guess(["D1", "D2", "D3", "D4"]))

    print("Game Solved?", game.is_solved())
    print("Game Lost?", game.is_lost())
    print("\nGame Summary:")
    import json
    print(json.dumps(game.get_summary(), indent=2))

    # Test max attempts
    game_lose = Game(test_puzzle)
    for i in range(Game.MAX_ATTEMPTS + 2):
         # Make deliberately bad guesses
         words = list(game_lose.get_remaining_words())
         if len(words) < 4: break
         guess = words[:3] + [words[-1]] # Likely incorrect/one away
         print(f"Attempt {i+1}: Guessing {guess}")
         result = game_lose.submit_guess(guess)
         print(f"Result: {result}")
         if game_lose.is_solved() or game_lose.is_lost() or result[0] == "LOST":
             break
    print("\nLosing Game Summary:")
    print(json.dumps(game_lose.get_summary(), indent=2)) 