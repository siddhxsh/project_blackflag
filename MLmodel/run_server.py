#!/usr/bin/env python
"""
Simple Flask server runner with environment setup
"""
import os
import sys

if __name__ == '__main__':
    # Ensure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Import and run the app
    from app import app
    
    # Get port from environment or default to 5000
    port = int(os.environ.get('PORT', 5000))
    
    print("\n" + "="*60)
    print(f"Starting Flask server on http://127.0.0.1:{port}")
    print("Press CTRL+C to stop")
    print("="*60 + "\n")
    
    # Run the app
    app.run(host='0.0.0.0', port=port, debug=False)
