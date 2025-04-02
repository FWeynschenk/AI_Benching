# NYT Connections AI Player

This project automatically fetches the daily NYT Connections puzzle answers, uses a configurable AI model (via Gemini or OpenAI APIs) to simulate playing the game based on the rules, logs the process, and posts the results to Bluesky.

## Architecture

The application is structured into several Python modules:

-   `main.py`: Main entry point, handles arguments, scheduling, and orchestrates the workflow.
-   `connections_fetcher.py`: Fetches puzzle data from the source URL.
-   `game_logic.py`: Defines the `Game` class to manage puzzle state and logic.
-   `ai_player.py`: Defines a base `BaseAIPlayer` and implementations (`GeminiPlayer`, `OpenAIPlayer`) to interact with the selected AI API model.
-   `bluesky_poster.py`: Handles formatting and posting results to Bluesky.
-   `logger_setup.py`: Configures logging for general events and detailed AI conversations.
-   `config.py`: Loads configuration (API keys) from environment variables.
-   `requirements.txt`: Lists Python dependencies (including `google-generativeai` and `openai`).
-   `.env`: Stores sensitive API keys and credentials (not committed to Git).
-   `connections_log.log`: Log file for general game events (INFO level, not committed to Git).
-   `conversation_log.log`: Detailed log of prompts sent to and responses received from the AI (DEBUG level, not committed to Git).

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    Create a `.env` file in the project root directory by copying `.env.example`:
    ```bash
    cp .env.example .env
    ```
    Edit the `.env` file and add your API keys and credentials:
    ```
    # Needed for Gemini models
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    
    # Needed for OpenAI models (gpt-*, etc.)
    OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
    
    # Needed for posting results
    BLUESKY_HANDLE="YOUR_BLUESKY_HANDLE"
    BLUESKY_PASSWORD="YOUR_BLUESKY_APP_PASSWORD" # Use an App Password!
    ```
    *(You only need to provide the API key for the service(s) you intend to use.)*

## Usage

By default, the script runs once on startup using the default AI model (`gemini-1.5-pro-latest`), posts the results to Bluesky, and then schedules itself to run daily at 7:00 AM local time.

**Running the Script:**

*   **Run with Scheduling (Default Model, Posting Enabled):**
    ```bash
    python main.py
    ```
    The script will run once immediately and then wait for the scheduled time. Press Ctrl+C to stop.

*   **Run Once (Default Model, Posting Enabled):**
    ```bash
    python main.py --run-once
    ```

*   **Run Once (Specific Model, Posting Disabled):**
    Use the `--model` argument to specify which AI model to use and `--no-post` to disable posting.
    ```bash
    python main.py --run-once --model [model_name] --no-post
    ```
    Replace `[model_name]` with one of the allowed models. The list of currently supported models and their providers is defined in the `SUPPORTED_MODELS` dictionary within `ai_player.py`. The choices presented by `--help` are derived from this dictionary.
    *(Ensure you have the corresponding API key configured in `.env` for the chosen model type.)*

*   **Run with Scheduling (Specific Model, Posting Enabled by default, can use `--post` explicitly):**
    ```bash
    python main.py --model [model_name]
    # or explicitly
    python main.py --model [model_name] --post 
    ```

*   **Controlling Bluesky Posting:**
    -   By default, results **are posted** to Bluesky.
    -   Use the `--no-post` flag to prevent posting:
        ```bash
        python main.py --run-once --no-post 
        ```
    -   Use the `--post` flag to explicitly enable posting (this is the default, so often not needed):
        ```bash
        python main.py --run-once --post
        ```

**Output:**

-   The script logs general progress and results to the console and `connections_log.log`.
-   Detailed AI prompts and responses are logged to `conversation_log.log`.
-   A summary post with the results and attempt grid is sent to the configured Bluesky account (unless `--no-post` is used).
-   A `results.json` file is created/updated in the root directory, storing a history of puzzle attempts (date, puzzle ID, model used, success status, attempts made).

**External Scheduling:**

For more robust scheduling, you can use your operating system's task scheduler instead of the built-in Python scheduler. Configure it to run the script with the `--run-once` flag and any desired `--model` or `--no-post` flags daily.

-   **Windows:** Task Scheduler
    Example command (using an OpenAI model, disabling posting):
    `C:\path\to\your\project\venv\Scripts\python.exe C:\path\to\your\project\main.py --run-once --model gpt-4o --no-post`
-   **macOS/Linux:** cron
    Example crontab entry (adjust path and time, using a Gemini model, default posting):
    `0 7 * * * /path/to/your/project/venv/bin/python /path/to/your/project/main.py --run-once --model gemini-1.5-pro-latest >> /path/to/your/project/cron.log 2>&1`

Remember to use absolute paths when configuring external schedulers. 