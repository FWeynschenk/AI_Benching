import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API keys and credentials
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BLUESKY_HANDLE = os.getenv("BLUESKY_HANDLE")
BLUESKY_PASSWORD = os.getenv("BLUESKY_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Basic validation
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found in environment variables.")
    # Consider raising an exception or exiting

if not BLUESKY_HANDLE or not BLUESKY_PASSWORD:
    print("Error: Bluesky handle or password not found in environment variables.")
    # Consider raising an exception or exiting 