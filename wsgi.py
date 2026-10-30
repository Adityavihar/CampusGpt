"""
Production WSGI entry point for CampusGPT.
Usage:
    - Linux: gunicorn -w 4 -b 0.0.0.0:5000 wsgi:app
    - Windows: waitress-serve --port=5000 wsgi:app
"""
import os
from app import app

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
