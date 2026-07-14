"""
run_production.py — Waitress WSGI production entry point for Windows IIS deployment.
This script dynamically imports the Flask app from 8_api.py to bypass digit-prefix
import restrictions and serves it using Waitress.
"""
import importlib.util
import os
import sys

# Ensure the working directory is the project root so relative paths (e.g. "8_api.py")
# resolve correctly regardless of how the process is launched. IIS HttpPlatformHandler
# starts the process from an arbitrary directory (often System32), so without this the
# dynamic import of "8_api.py" below would fail.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Configure environment variables before importing
os.environ.setdefault("PORT", "5000")
os.environ.setdefault("HOST", "127.0.0.1")
# Ensure lazy loading is enabled in production to prevent startup timeouts
os.environ.setdefault("V2_LAZY_LOAD", "1")

# Import the 8_api.py module dynamically
print("Loading Address AI API...")
spec = importlib.util.spec_from_file_location("api_module", "8_api.py")
api_module = importlib.util.module_from_spec(spec)
sys.modules["api_module"] = api_module
spec.loader.exec_module(api_module)

app = api_module.app

# Try importing waitress, fallback with instructions if not installed
try:
    from waitress import serve
except ImportError:
    print("\n[!] Waitress is not installed. Please run: pip install waitress\n")
    sys.exit(1)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    host = os.getenv("HOST", "127.0.0.1")
    print("=" * 60)
    print(f"  Starting Waitress Production Server")
    print(f"  Running on http://{host}:{port}/")
    print("  To stop: Press CTRL+C")
    print("=" * 60)
    
    # Run the Waitress WSGI server (configured with 8 worker threads for high concurrency)
    serve(app, host=host, port=port, threads=8)
