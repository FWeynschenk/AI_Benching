import logging
import time
import re
import json
from abc import ABC, abstractmethod # For base class

# API Clients
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from openai import OpenAI, RateLimitError, APIError # Import OpenAI client and specific errors
import anthropic # Import Anthropic client

# Config
from config import GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY

logger = logging.getLogger(__name__)
conv_logger = logging.getLogger("conversation")

# --- Base Class --- 
class BaseAIPlayer(ABC):
    """Abstract base class for different AI model players."""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._configure_client()
        self._initialize_model()

    @abstractmethod
    def _configure_client(self):
        """Configure the specific API client (Gemini, OpenAI, etc.)."""
        pass

    @abstractmethod
    def _initialize_model(self):
        """Initialize the specific model instance for the player."""
        pass

    @abstractmethod
    def get_ai_guess(self, remaining_words: list[str], attempt_history: list, max_retries=3) -> list[str] | None:
        """Gets a guess from the AI.

        Args:
            remaining_words: The list of words currently available.
            attempt_history: List of previous attempts (tuples from Game.log).
            max_retries: Maximum number of times to retry API call or parsing.

        Returns:
            A list of four guessed words, or None if unable to get a valid guess.
        """
        pass

    def _build_prompt(self, remaining_words: list[str], attempt_history: list) -> str:
        """Builds the core text prompt (can be reused/adapted by subclasses)."""
        # This prompt structure seems generally applicable
        prompt = f"You are playing the NYT Connections puzzle game. "
        prompt += "The goal is to find groups of four words that share a common theme or category.\n Beware that the creator of the puzzle will try to trick you. If five words are related to each other, they can't all be in the same group.\n\n"
        prompt += "Here are the rules:\n"
        prompt += "1. Look at the available words and try to identify potential connections.\n"
        prompt += "2. Select exactly four words you believe belong together in a single group.\n"
        prompt += "3. Submit the guess. If correct, the words are removed. If incorrect, you try again.\n"
        prompt += "4. Feedback might be given if a guess is 'one away' (three words belong to a correct group).\n\n"

        prompt += "Available words:\n"
        prompt += ", ".join(sorted(remaining_words))
        prompt += "\n\n"

        if attempt_history:
            prompt += "Previous attempts:\n"
            for i, attempt in enumerate(attempt_history):
                num, words, status, category, color = attempt
                words_str = ", ".join(words)
                if status == "CORRECT":
                    prompt += f"- Attempt {num}: {words_str} -> CORRECT\n"
                elif status == "ONE_AWAY":
                    prompt += f"- Attempt {num}: {words_str} -> INCORRECT (Hint: One Away)\n"
                elif status == "INCORRECT":
                     prompt += f"- Attempt {num}: {words_str} -> INCORRECT\n"
                elif status == "ALREADY_FOUND":
                     prompt += f"- Attempt {num}: {words_str} -> Already Found\n"
            prompt += "\n"

        prompt += "Based on the available words and previous attempts, please identify one group of four related words. "
        prompt += "Return ONLY the four words, separated by commas. For example: WORD1, WORD2, WORD3, WORD4"

        return prompt

    def _parse_response(self, response_text: str, valid_words: set) -> list[str] | None:
        """Parses the AI response to extract four valid words (reusable)."""
        conv_logger.debug(f"<<< Raw AI Response:\n{response_text}")
        logger.debug(f"Raw AI Response: {response_text}")

        match = re.search(r"([\w\s'/-]+,\s*[\w\s'/-]+,\s*[\w\s'/-]+,\s*[\w\s'/-]+)", response_text)

        if match:
            potential_words_str = match.group(1)
            guessed_words_raw = [word.strip().upper() for word in potential_words_str.split(',')]
            guessed_words = [word for word in guessed_words_raw if word]
            logger.debug(f"Parsed words (initial): {guessed_words}")

            if len(guessed_words) == 4:
                valid_set_upper = {word.upper() for word in valid_words}
                if all(word in valid_set_upper for word in guessed_words):
                    logger.info(f"AI suggested valid group: {guessed_words}")
                    return guessed_words
                else:
                    invalid = [word for word in guessed_words if word not in valid_set_upper]
                    logger.warning(f"AI response contained invalid/used words: {invalid}. Parsed: {guessed_words}. Valid: {valid_set_upper}")
                    return None
            else:
                 logger.warning(f"AI response parsing did not yield exactly 4 words. Found {len(guessed_words)}: {guessed_words}")
                 return None
        else:
            logger.warning(f"Could not parse 4 comma-separated words from AI response: {response_text}")
            return None

    @abstractmethod
    def get_category_guesses(self, found_groups: list[dict]) -> dict[str, str]:
        """Gets the AI's category guesses for each found group.

        Args:
            found_groups: List of dictionaries containing group details (color, members, actual category)

        Returns:
            Dictionary mapping color to AI's guessed category name
        """
        pass

# --- Gemini Player --- 
class GeminiPlayer(BaseAIPlayer):
    """AI Player implementation using the Google Gemini API."""
    
    def _configure_client(self):
        if not GEMINI_API_KEY:
            logger.error("Gemini API Key not configured.")
            raise ValueError("GEMINI_API_KEY is not set for GeminiPlayer.")
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            logger.info("Gemini API client configured.")
        except Exception as e:
            logger.error(f"Failed to configure Gemini API: {e}")
            raise

    def _initialize_model(self):
        logger.info(f"Initializing GeminiPlayer with model: {self.model_name}")
        self.generation_config = { "temperature": 0.7, "top_p": 1, "top_k": 1 }
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        try:
            self.model = genai.GenerativeModel(model_name=self.model_name,
                                               generation_config=self.generation_config,
                                               safety_settings=self.safety_settings)
            logger.info(f"Gemini model '{self.model_name}' initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model '{self.model_name}': {e}")
            raise
            
    def get_ai_guess(self, remaining_words: list[str], attempt_history: list, max_retries=5) -> list[str] | None:
        if not remaining_words or len(remaining_words) < 4:
            logger.error("Not enough remaining words for AI to make a guess.")
            return None

        prompt = self._build_prompt(remaining_words, attempt_history)
        conv_logger.debug(f">>> [Gemini] Prompt Sent:\n{prompt}")
        logger.debug(f"Generated Prompt:\n{prompt}")

        valid_word_set = set(remaining_words)

        for attempt in range(max_retries):
            try:
                logger.info(f"Sending request to Gemini API (Attempt {attempt + 1}/{max_retries})...")
                response = self.model.generate_content(prompt)
                # First check if response was blocked by safety settings
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    logger.error(f"Prompt blocked by safety settings. Reason: {response.prompt_feedback.block_reason}")
                    return None

                # Check for empty candidates explicitly
                if not response.candidates:
                    logger.warning("Gemini response has no candidates (empty response)")
                    # --- MODIFIED LOGGING ---
                    # Try to access finish reason and safety ratings from the underlying proto result
                    finish_reason = "UNKNOWN"
                    safety_ratings = "UNKNOWN"
                    try:
                        if hasattr(response, '_result') and response._result:
                           # Access fields directly from the protos.GenerateContentResponse
                           # Adjust field names based on actual protobuf definition if necessary
                           finish_reason = getattr(response._result, 'finish_reason', "NOT_FOUND") 
                           safety_ratings = getattr(response._result, 'safety_ratings', "NOT_FOUND")
                           # Also log the prompt feedback for completeness here
                           prompt_feedback_info = getattr(response, 'prompt_feedback', "NOT_FOUND")
                        else:
                           prompt_feedback_info = getattr(response, 'prompt_feedback', "NOT_FOUND")

                        logger.warning(f"Empty candidates details: FinishReason={finish_reason}, SafetyRatings={safety_ratings}, PromptFeedback={prompt_feedback_info}")

                        # --- ADDED FOR DEEPER INSPECTION ---
                        try:
                           if hasattr(response, '_result') and response._result:
                              logger.debug(f"Attributes of response._result: {dir(response._result)}")
                              logger.debug(f"String representation of response._result: {str(response._result)}")
                           else:
                              logger.debug("response._result object not found or is empty.")
                        except Exception as inspect_e:
                           logger.error(f"Error during deeper inspection of response._result: {inspect_e}")
                        # --- END DEEPER INSPECTION ---

                    except Exception as log_e:
                         logger.error(f"Error trying to log detailed empty response info: {log_e}")
                    # --- END MODIFIED LOGGING ---

                    if attempt < max_retries - 1:
                        wait_time = (3 ** attempt) + 5
                        logger.info(f"Empty candidates - waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error("Empty candidates on final attempt. Giving up.")
                        return None

                if not response.parts:
                    logger.warning(f"Gemini response has no parts/empty candidates. Response: {response}")
                    # This could be due to rate limiting - use exponential backoff
                    if attempt < max_retries - 1:
                        wait_time = (3 ** attempt) + 5
                        logger.info(f"Empty response (possible rate limit) - waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error("Empty response on final attempt. Giving up.")
                        return None

                response_text = response.text
                parsed_guess = self._parse_response(response_text, valid_word_set)

                if parsed_guess:
                    return parsed_guess
                else:
                    logger.warning(f"Failed to parse valid group from Gemini response on attempt {attempt + 1}. Retrying...")
                    conv_logger.debug(f"--- Failed to parse Gemini response on attempt {attempt + 1} ---")
                    prompt += "\n\nPlease ensure you return exactly four words from the list, separated only by commas."
                    wait_time = (3 ** attempt) + 5
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)

            except ResourceExhausted as e: 
                logger.warning(f"Gemini API rate limit hit on attempt {attempt + 1}: {e}")
                retry_delay_match = re.search(r"retry_delay {\s*seconds: (\d+)\s*}", str(e))
                wait_time = 0
                if retry_delay_match:
                    wait_time = int(retry_delay_match.group(1)) + 5  # Add 5s buffer instead of 1s
                    logger.info(f"API suggests waiting {wait_time -5}s. Waiting {wait_time}s before retry...")
                else:
                    wait_time = (3 ** attempt) + 5
                    logger.warning(f"Could not parse retry_delay from message. Using exponential backoff: waiting {wait_time}s...")
                
                if attempt < max_retries - 1: 
                    time.sleep(wait_time)
                else:
                    logger.error(f"Rate limit hit on final attempt ({attempt + 1}). Giving up.")

            except Exception as e:
                logger.error(f"Error calling Gemini API or processing response on attempt {attempt + 1}: {e}", exc_info=True)
                if attempt < max_retries - 1:
                     wait_time = (3 ** attempt) + 5
                     logger.info(f"Waiting {wait_time}s before retry due to unexpected error...")
                     time.sleep(wait_time)
                else:
                    logger.error(f"Unexpected error on final attempt ({attempt + 1}). Giving up.")

        logger.error(f"Failed to get a valid guess from Gemini AI after {max_retries} attempts.")
        conv_logger.error(f"--- Failed to get valid guess from Gemini after {max_retries} attempts --- ")
        return None

    def get_category_guesses(self, found_groups: list[dict]) -> dict[str, str]:
        """Gets Gemini's category guesses for each found group."""
        prompt = "You just successfully solved a Connections puzzle! For each group of words you found, please provide a short category name that describes what they have in common.\n\n"
        
        for group in found_groups:
            color = group['color']
            words = group['members']
            prompt += f"{color} group words: {', '.join(words)}\n"
        
        prompt += "\nPlease provide your category guesses in the format:\n"
        prompt += "Yellow: [your guess]\n"
        prompt += "Green: [your guess]\n"
        prompt += "Blue: [your guess]\n"
        prompt += "Purple: [your guess]\n"

        for attempt in range(3):  # Try up to 3 times
            try:
                response = self.model.generate_content(prompt)
                response_text = response.text.strip()
                
                # Parse the response to extract category guesses
                category_guesses = {}
                for line in response_text.split('\n'):
                    if ':' in line:
                        color, guess = line.split(':', 1)
                        color = color.strip()
                        guess = guess.strip()
                        if color in ['Yellow', 'Green', 'Blue', 'Purple']:
                            category_guesses[color] = guess
                
                # Validate we got guesses for all colors
                if len(category_guesses) == 4:
                    return category_guesses
                else:
                    logger.warning(f"Gemini category guesses incomplete on attempt {attempt + 1}. Got {len(category_guesses)}/4 guesses.")
                    if attempt < 2:  # If not last attempt
                        wait_time = (3 ** attempt) + 5  # Exponential backoff: 5s, 14s, 32s
                        logger.info(f"Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error("Failed to get complete category guesses after 3 attempts")
                        return category_guesses  # Return partial results if any

            except ResourceExhausted as e:
                logger.warning(f"Gemini API rate limit hit on attempt {attempt + 1}: {e}")
                if attempt < 2:
                    wait_time = (3 ** attempt) + 5
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error("Rate limit hit on final attempt. Giving up.")
                    return {}

            except Exception as e:
                logger.error(f"Error getting category guesses from Gemini on attempt {attempt + 1}: {e}")
                if attempt < 2:
                    wait_time = (3 ** attempt) + 5
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error("Failed to get category guesses after 3 attempts")
                    return {}

        return {}  # Should never reach here, but just in case

# --- OpenAI Player --- 
class OpenAIPlayer(BaseAIPlayer):
    """AI Player implementation using the OpenAI API."""

    def _configure_client(self):
        if not OPENAI_API_KEY:
            logger.error("OpenAI API Key not configured.")
            raise ValueError("OPENAI_API_KEY is not set for OpenAIPlayer.")
        try:
            # Client is initialized in _initialize_model as it doesn't have a global configure step like Gemini
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            logger.info("OpenAI API client configured (key ready).")
        except Exception as e:
            logger.error(f"Failed to prepare OpenAI client configuration: {e}")
            raise

    def _initialize_model(self):
        # For OpenAI, model initialization is mainly just storing the name.
        # Actual model is specified during the API call.
        logger.info(f"Initializing OpenAIPlayer with model: {self.model_name}")
        # No specific model object to create here, just ensure client is ready
        if not hasattr(self, 'client') or not self.client:
             raise RuntimeError("OpenAI client was not configured before model initialization.")
        pass # Model is ready to use via self.client

    def get_ai_guess(self, remaining_words: list[str], attempt_history: list, max_retries=3) -> list[str] | None:
        if not remaining_words or len(remaining_words) < 4:
            logger.error("Not enough remaining words for AI to make a guess.")
            return None

        # Build prompt for chat model (system prompt + user prompt)
        system_prompt = "You are playing the NYT Connections puzzle game. The goal is to find groups of four words that share a common theme or category. Return ONLY the four words, separated by commas. For example: WORD1, WORD2, WORD3, WORD4"
        user_prompt = self._build_prompt(remaining_words, attempt_history) # Reuse core prompt logic
        
        # Remove the instruction part from the user prompt as it's in the system prompt now
        user_prompt = user_prompt.split("Return ONLY the four words")[0].strip()

        messages = [
            # {"role": "system", "content": system_prompt},
            {"role": "user", "content": system_prompt + "\n\n" + user_prompt}
        ]

        conv_logger.debug(f">>> [OpenAI] Messages Sent:\n{json.dumps(messages, indent=2)}")
        logger.debug(f"Generated Prompt (User part):\n{user_prompt}")

        valid_word_set = set(remaining_words)

        for attempt in range(max_retries):
            try:
                logger.info(f"Sending request to OpenAI API (Model: {self.model_name}, Attempt {attempt + 1}/{max_retries})...")
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                )
                
                response_text = response.choices[0].message.content.strip()
                parsed_guess = self._parse_response(response_text, valid_word_set)

                if parsed_guess:
                    return parsed_guess
                else:
                    logger.warning(f"Failed to parse valid group from OpenAI response on attempt {attempt + 1}. Retrying...")
                    conv_logger.debug(f"--- Failed to parse OpenAI response on attempt {attempt + 1} ---")
                    # Add a reminder message for the next attempt?
                    messages.append({"role": "assistant", "content": response_text}) # Add previous response
                    messages.append({"role": "user", "content": "That wasn't quite right. Please remember to return exactly four words from the list, separated only by commas."})
                    time.sleep(1)

            except RateLimitError as e:
                logger.warning(f"OpenAI API rate limit hit on attempt {attempt + 1}: {e}")
                # OpenAI library might handle retries automatically or provide headers
                # For now, use simple exponential backoff
                wait_time = (2 ** attempt) + 1
                logger.info(f"Waiting {wait_time}s before retry due to rate limit...")
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                else:
                    logger.error(f"Rate limit hit on final attempt ({attempt + 1}). Giving up.")
            
            except APIError as e:
                 logger.error(f"OpenAI API error on attempt {attempt + 1}: {e}", exc_info=True)
                 # Use exponential backoff for general API errors
                 if attempt < max_retries - 1:
                     wait_time = 2 ** attempt
                     logger.info(f"Waiting {wait_time}s before retry due to API error...")
                     time.sleep(wait_time)
                 else:
                    logger.error(f"API error on final attempt ({attempt + 1}). Giving up.")

            except Exception as e:
                logger.error(f"Error calling OpenAI API or processing response on attempt {attempt + 1}: {e}", exc_info=True)
                # Use exponential backoff for other unexpected errors
                if attempt < max_retries - 1:
                     wait_time = 2 ** attempt
                     logger.info(f"Waiting {wait_time}s before retry due to unexpected error...")
                     time.sleep(wait_time)
                else:
                    logger.error(f"Unexpected error on final attempt ({attempt + 1}). Giving up.")

        logger.error(f"Failed to get a valid guess from OpenAI AI after {max_retries} attempts.")
        conv_logger.error(f"--- Failed to get valid guess from OpenAI after {max_retries} attempts --- ")
        return None

    def get_category_guesses(self, found_groups: list[dict]) -> dict[str, str]:
        """Gets OpenAI's category guesses for each found group."""
        prompt = "This is the result of a NYT Connections puzzle. For each group of words found, provide a short category name that describes what they exclusively have in common, the category of a group should not apply to words in one of the other groups.\n\n"
        
        for group in found_groups:
            color = group['color']
            words = group['members']
            prompt += f"{color} group words: {', '.join(words)}\n"
        
        prompt += "\nPlease provide your category guesses in the format:\n"
        prompt += "Yellow: [your guess]\n"
        prompt += "Green: [your guess]\n"
        prompt += "Blue: [your guess]\n"
        prompt += "Purple: [your guess]\n"

        for attempt in range(3):  # Try up to 3 times
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}]
                )
                response_text = response.choices[0].message.content.strip()
                
                # Parse the response to extract category guesses
                category_guesses = {}
                for line in response_text.split('\n'):
                    if ':' in line:
                        color, guess = line.split(':', 1)
                        color = color.strip()
                        guess = guess.strip()
                        if color in ['Yellow', 'Green', 'Blue', 'Purple']:
                            category_guesses[color] = guess
                
                # Validate we got guesses for all colors
                if len(category_guesses) == 4:
                    return category_guesses
                else:
                    logger.warning(f"OpenAI category guesses incomplete on attempt {attempt + 1}. Got {len(category_guesses)}/4 guesses.")
                    if attempt < 2:  # If not last attempt
                        wait_time = (2 ** attempt) + 1  # Exponential backoff: 1s, 3s, 5s
                        logger.info(f"Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error("Failed to get complete category guesses after 3 attempts")
                        return category_guesses  # Return partial results if any

            except RateLimitError as e:
                logger.warning(f"OpenAI API rate limit hit on attempt {attempt + 1}: {e}")
                if attempt < 2:
                    wait_time = (2 ** attempt) + 1
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error("Rate limit hit on final attempt. Giving up.")
                    return {}

            except APIError as e:
                logger.error(f"OpenAI API error on attempt {attempt + 1}: {e}")
                if attempt < 2:
                    wait_time = 2 ** attempt
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error("API error on final attempt. Giving up.")
                    return {}

            except Exception as e:
                logger.error(f"Error getting category guesses from OpenAI on attempt {attempt + 1}: {e}")
                if attempt < 2:
                    wait_time = 2 ** attempt
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error("Failed to get category guesses after 3 attempts")
                    return {}

        return {}  # Should never reach here, but just in case

# --- Anthropic Player --- 
class AnthropicPlayer(BaseAIPlayer):
    """AI Player implementation using the Anthropic API."""

    def _configure_client(self):
        if not ANTHROPIC_API_KEY:
            logger.error("Anthropic API Key not configured.")
            raise ValueError("ANTHROPIC_API_KEY is not set for AnthropicPlayer.")
        try:
            self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            logger.info("Anthropic API client configured.")
        except Exception as e:
            logger.error(f"Failed to configure Anthropic API: {e}")
            raise

    def _initialize_model(self):
        logger.info(f"Initializing AnthropicPlayer with model: {self.model_name}")
        if not hasattr(self, 'client') or not self.client:
            raise RuntimeError("Anthropic client was not configured before model initialization.")
        pass # Model is ready to use via self.client

    def get_ai_guess(self, remaining_words: list[str], attempt_history: list, max_retries=3) -> list[str] | None:
        if not remaining_words or len(remaining_words) < 4:
            logger.error("Not enough remaining words for a guess.")
            return None

        # Create a set of valid words for validation
        valid_word_set = set(word.upper() for word in remaining_words)

        # Build the prompt
        prompt = self._build_prompt(remaining_words, attempt_history)

        # Prepare messages for Anthropic
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        for attempt in range(max_retries):
            try:
                logger.info(f"Sending request to Anthropic API (Model: {self.model_name}, Attempt {attempt + 1}/{max_retries})...")
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=1024,
                    messages=messages,
                    temperature=0.7
                )
                
                response_text = response.content[0].text.strip()
                parsed_guess = self._parse_response(response_text, valid_word_set)

                if parsed_guess:
                    return parsed_guess
                else:
                    logger.warning(f"Failed to parse valid group from Anthropic response on attempt {attempt + 1}. Retrying...")
                    conv_logger.debug(f"--- Failed to parse Anthropic response on attempt {attempt + 1} ---")
                    # Add a reminder message for the next attempt
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append({"role": "user", "content": "That wasn't quite right. Please remember to return exactly four words from the list, separated only by commas."})
                    time.sleep(1)

            except Exception as e:
                logger.warning(f"Anthropic API error on attempt {attempt + 1}: {e}")
                wait_time = (2 ** attempt) + 1
                logger.info(f"Waiting {wait_time}s before retry...")
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                else:
                    logger.error(f"API error on final attempt ({attempt + 1}). Giving up.")
                    return None

        return None

    def get_category_guesses(self, found_groups: list[dict]) -> dict[str, str]:
        """Gets Anthropic's category guesses for each found group."""
        prompt = "You just successfully solved a Connections puzzle! For each group of words you found, please provide a short category name that describes what they have in common.\n\n"
        
        for group in found_groups:
            color = group['color']
            words = group['members']
            prompt += f"{color} group words: {', '.join(words)}\n"
        
        prompt += "\nPlease provide your category guesses in the format:\n"
        prompt += "Yellow: [your guess]\n"
        prompt += "Green: [your guess]\n"
        prompt += "Blue: [your guess]\n"
        prompt += "Purple: [your guess]\n"

        for attempt in range(3):  # Try up to 3 times
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )
                response_text = response.content[0].text.strip()
                
                # Parse the response to extract category guesses
                category_guesses = {}
                for line in response_text.split('\n'):
                    if ':' in line:
                        color, guess = line.split(':', 1)
                        color = color.strip()
                        guess = guess.strip()
                        if color in ['Yellow', 'Green', 'Blue', 'Purple']:
                            category_guesses[color] = guess
                
                # Validate we got guesses for all colors
                if len(category_guesses) == 4:
                    return category_guesses
                else:
                    logger.warning(f"Anthropic category guesses incomplete on attempt {attempt + 1}. Got {len(category_guesses)}/4 guesses.")
                    if attempt < 2:  # If not last attempt
                        wait_time = (2 ** attempt) + 1  # Exponential backoff: 1s, 3s, 5s
                        logger.info(f"Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error("Failed to get complete category guesses after 3 attempts")
                        return category_guesses  # Return partial results if any

            except Exception as e:
                logger.error(f"Error getting category guesses from Anthropic on attempt {attempt + 1}: {e}")
                if attempt < 2:
                    wait_time = (2 ** attempt) + 1
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error("Failed to get category guesses after 3 attempts")
                    return {}

        return {}  # Should never reach here, but just in case

# --- Model Provider Mapping --- 
SUPPORTED_MODELS = {
    # Gemini
    "gemini-1.5-pro": "gemini",
    "gemini-1.5-flash": "gemini",
    "gemini-1.5-flash-8b": "gemini",
    "gemini-2.0-flash": "gemini",
    "gemini-2.0-flash-lite": "gemini",
    "gemini-2.5-pro-preview-05-06": "gemini",
    "gemini-2.5-pro-preview-06-05": "gemini",
    "gemini-2.5-pro-exp-03-25": "gemini",
    "gemini-2.5-flash-preview-05-20": "gemini",
    # OpenAI
    "o4-mini": "openai",
    "o3-mini": "openai",
    "o1-mini": "openai",
    "o1-pro": "openai",
    "o1": "openai",
    "o3": "openai",
    "gpt-4o-mini": "openai",
    "gpt-4o": "openai",
    "gpt-4.5-preview": "openai",
    # Anthropic
    "claude-3-7-sonnet-20250219": "anthropic",
    "claude-3-5-sonnet-20241022": "anthropic",
    "claude-3-5-haiku-20241022": "anthropic",
    "claude-3-opus-20240229": "anthropic",
    "claude-sonnet-4-20250514": "anthropic",
}

# --- Factory Function (Updated) --- 
def get_ai_player(model_name: str) -> BaseAIPlayer:
    """Factory function to create the correct AI player based on model name using a dictionary lookup."""
    # Normalize the input model name slightly (lowercase, remove potential openai/ prefix)
    # This helps match keys in the dictionary if user uses slightly different casing or includes prefix
    normalized_name_for_lookup = model_name.lower().split('/')[-1]
    
    # Try lookup with the normalized name first
    provider = SUPPORTED_MODELS.get(normalized_name_for_lookup)
    
    # If not found, try the original name (in case it had a different prefix or structure)
    if provider is None:
        provider = SUPPORTED_MODELS.get(model_name)

    if provider == "gemini":
        # Pass the original model_name to the constructor
        return GeminiPlayer(model_name)
    elif provider == "openai":
        # Pass the potentially normalized name (without openai/ prefix) to constructor
        # OpenAI player expects the actual model identifier (e.g., 'gpt-4o')
        actual_model_name = model_name.split('/')[-1] 
        return OpenAIPlayer(actual_model_name)
    elif provider == "anthropic":
        # Pass the potentially normalized name to constructor
        actual_model_name = model_name.split('/')[-1]
        return AnthropicPlayer(actual_model_name)
    else:
        logger.error(f"Unsupported model name: {model_name}. Not found in SUPPORTED_MODELS map.")
        # Suggest available models
        available = list(SUPPORTED_MODELS.keys())
        # Filter out prefixed versions for suggestion to avoid confusion
        available_clean = sorted([m for m in available if not m.startswith("openai/")])
        logger.info(f"Available models: {available_clean}")
        raise ValueError(f"Unsupported AI model type: {model_name}")

# Example Usage (for testing)
if __name__ == "__main__":
    from logger_setup import setup_logging
    setup_logging()
    from main import DEFAULT_MODEL # Get default model for testing

    # Ensure you have a .env file with API keys set
    try:
        # Test Gemini Player
        print("--- Testing Gemini Player ---")
        gemini_player = get_ai_player(DEFAULT_MODEL)
        # ... (Add test logic using gemini_player.get_ai_guess) ...
        print(f"Gemini Player initialized with {gemini_player.model_name}")

        # Test OpenAI Player (if key exists and model is specified)
        # Example: Test with gpt-4o if key is available
        if OPENAI_API_KEY:
             print("\n--- Testing OpenAI Player (gpt-4o) ---")
             try:
                 # Use a known valid OpenAI model identifier
                 openai_player = get_ai_player("gpt-4o") 
                 # ... (Add test logic using openai_player.get_ai_guess) ...
                 print(f"OpenAI Player initialized with {openai_player.model_name}")
             except Exception as e:
                  print(f"Could not initialize or test OpenAI player: {e}")
        else:
            print("\nSkipping OpenAI Player test: OPENAI_API_KEY not set.")

    except ValueError as e:
        print(f"Setup Error: {e}")
    except Exception as e:
         print(f"An unexpected error occurred during testing: {e}") 
