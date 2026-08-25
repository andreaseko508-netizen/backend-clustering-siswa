import uvicorn
import os
import sys

# Add the current directory to sys.path so it can find api package
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from api.index import app

if __name__ == "__main__":
    # Local Development Entry Point
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting SIMORBATAS Backend on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
