"""
API entry point for Render deployment.
This module imports the Flask app from the parent app.py file.
"""

import sys
import os

# Add parent directory to path to import app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

# Make app available for gunicorn
if __name__ == '__main__':
    app.run(debug=False)
