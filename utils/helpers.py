"""
Helper Utilities
Common utility functions used across the project.
"""

import logging
import os
import sys
import json
from datetime import datetime


def setup_logging(log_file=None, log_level="INFO"):
    """
    Configure application-wide logging.
    
    Args:
        log_file: Path to log file (optional)
        log_level: Logging level string
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    logging.info("Logging system initialized")


def load_json(path):
    """Load a JSON file and return its contents."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"JSON file not found: {path}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"JSON parse error in {path}: {e}")
        return None


def save_json(data, path):
    """Save data to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logging.info(f"Data saved to {path}")


def get_timestamp():
    """Get current timestamp as formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dirs(*dirs):
    """Create directories if they don't exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)
