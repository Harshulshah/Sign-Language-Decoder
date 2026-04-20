"""
Sign Language Decoder — Main Application
A real-time AI-powered sign language recognition system.

Usage:
    python app.py              Start the Flask web server (local dev)
    gunicorn app:create_app()  Start with gunicorn (production)

Server runs at: http://127.0.0.1:5000 (local) or assigned port (cloud)
"""

import os
import sys
import logging
import argparse

from flask import Flask, render_template
from flask_cors import CORS

from config import (
    FLASK_HOST, FLASK_PORT, FLASK_DEBUG,
    STATIC_DIR, TEMPLATE_DIR, LOG_FILE, LOG_LEVEL,
    MODEL_PATH, MODEL_DIR, DATA_DIR,
)
from utils.helpers import setup_logging, ensure_dirs
from api.routes import api, init_components


def create_app():
    """Application factory: create and configure the Flask app."""
    app = Flask(
        __name__,
        static_folder=STATIC_DIR,
        template_folder=TEMPLATE_DIR,
    )
    CORS(app)

    # Register API blueprint
    app.register_blueprint(api, url_prefix="/api")

    # ── Page Routes ──────────────────────────────────────────
    @app.route("/")
    def index():
        return render_template("index.html")

    @app.errorhandler(404)
    def not_found(e):
        return render_template("index.html"), 404

    # Initialize components on first request or at startup
    with app.app_context():
        # Setup logging
        setup_logging(log_file=LOG_FILE, log_level=LOG_LEVEL)

        # Ensure directories exist
        ensure_dirs(MODEL_DIR, DATA_DIR, STATIC_DIR, TEMPLATE_DIR)

        # Generate demo model if needed
        if not os.path.exists(MODEL_PATH):
            try:
                logging.getLogger(__name__).info("No model found — generating demo model...")
                from model.train import generate_demo_model
                generate_demo_model()
            except Exception as e:
                logging.getLogger(__name__).warning(f"Demo model generation skipped: {e}")

        # Initialize detection components
        init_components()

    return app


def main():
    parser = argparse.ArgumentParser(description="Sign Language Decoder")
    parser.add_argument(
        "--generate", action="store_true",
        help="Generate a demo model before starting the server",
    )
    parser.add_argument(
        "--host", type=str, default=FLASK_HOST,
        help=f"Server host (default: {FLASK_HOST})",
    )
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("PORT", FLASK_PORT)),
        help=f"Server port (default: {FLASK_PORT})",
    )
    args = parser.parse_args()

    # Setup logging
    setup_logging(log_file=LOG_FILE, log_level=LOG_LEVEL)
    logger = logging.getLogger(__name__)

    # Ensure directories exist
    ensure_dirs(MODEL_DIR, DATA_DIR, STATIC_DIR, TEMPLATE_DIR)

    # Generate demo model if requested
    if args.generate or not os.path.exists(MODEL_PATH):
        logger.info("No model found — generating demo model...")
        print("\n" + "=" * 60)
        print("  Generating demo model (one-time setup)...")
        print("  This will take about 30 seconds.")
        print("=" * 60 + "\n")
        try:
            from model.train import generate_demo_model
            generate_demo_model()
            print("\n  Demo model generated successfully!\n")
        except Exception as e:
            logger.warning(f"Demo model generation skipped: {e}")

    # Initialize components
    init_components()

    # Create and run app
    app = create_app()

    print("\n" + "=" * 60)
    print(f"  Sign Language Decoder is running!")
    print(f"  Open: http://{args.host}:{args.port}")
    print(f"  Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    app.run(
        host=args.host,
        port=args.port,
        debug=False,
        threaded=True,
    )


if __name__ == "__main__":
    main()
