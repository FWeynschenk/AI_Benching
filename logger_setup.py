import logging
import sys

LOG_FILE = 'connections_log.log'
CONVERSATION_LOG_FILE = 'conversation_log.log'

def setup_logging():
    """Configures logging for general info and detailed AI conversation."""
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # === Root Logger (connections_log.log + Console) ===
    root_logger = logging.getLogger() # Get the root logger
    root_logger.setLevel(logging.INFO) # Set root level (INFO and above)

    # File Handler for root logger (connections_log.log)
    root_file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
    root_file_handler.setFormatter(log_formatter)
    root_logger.addHandler(root_file_handler)

    # Console Handler for root logger
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    try:
        console_handler.stream.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
    except Exception as e:
        logging.warning(f"Could not reconfigure stdout encoding to UTF-8: {e}", exc_info=False)
        pass
    root_logger.addHandler(console_handler)

    # === Conversation Logger (conversation_log.log) ===
    conv_logger = logging.getLogger("conversation")
    conv_logger.setLevel(logging.DEBUG) # Log DEBUG level for conversation details
    conv_logger.propagate = False # Prevent messages from going to root logger handlers

    # File Handler for conversation logger
    conv_file_handler = logging.FileHandler(CONVERSATION_LOG_FILE, mode='a', encoding='utf-8')
    # Use a simpler format for conversation log? Or same?
    # Let's use the same for consistency for now.
    conv_file_handler.setFormatter(log_formatter) 
    conv_logger.addHandler(conv_file_handler)

    # Note: We no longer use basicConfig as we are configuring handlers manually.
    
# If you want to get a logger instance elsewhere:
# logger = logging.getLogger(__name__) # For general logs
# conv_logger = logging.getLogger("conversation") # For conversation logs

# Call setup_logging() when the module is imported?
# Or perhaps call it explicitly in main.py
# Let's configure it here for simplicity for now.
# setup_logging()
# If you want to get a logger instance elsewhere:
# logger = logging.getLogger(__name__) 