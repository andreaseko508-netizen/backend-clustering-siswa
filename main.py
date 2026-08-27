import uvicorn
import os
import sys

# Ensure the root directory is in sys.path to resolve 'api' package
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from api.index import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Local Development Server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
