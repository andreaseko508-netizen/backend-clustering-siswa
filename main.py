import uvicorn
import os
import sys

# S2 Hardening: Add the current directory to sys.path to resolve the 'api' package locally
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from api.index import app

if __name__ == "__main__":
    # Local Development Entry Point (Demo/Testing)
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting SIMORBATAS Final Engine on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
