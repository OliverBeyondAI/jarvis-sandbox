"""
Prior Authorization Portal - Local HTTP Server

A simple Python server that serves the multi-step prior authorization form
and handles form submissions via a JSON API endpoint.

Usage:
    python server.py [--port PORT]

The server starts on http://localhost:8080 by default.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

SUBMISSIONS_DIR = Path(__file__).parent / "submissions"


class PARequestHandler(SimpleHTTPRequestHandler):
    """Handler that serves the static form and accepts JSON submissions."""

    def __init__(self, *args, **kwargs):
        # Serve files from the portal directory
        super().__init__(*args, directory=str(Path(__file__).parent), **kwargs)

    def do_POST(self):
        if self.path == "/api/submit":
            self._handle_submit()
        else:
            self.send_error(404, "Not Found")

    def _handle_submit(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        # Save submission to disk
        SUBMISSIONS_DIR.mkdir(exist_ok=True)
        ref = data.get("referenceNumber", "unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = SUBMISSIONS_DIR / f"{ref}_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

        print(f"[{datetime.now().isoformat()}] Submission saved: {filename.name}")

        # Return success response
        response = json.dumps({
            "status": "received",
            "referenceNumber": ref,
            "message": "Prior authorization request received and pending review.",
            "savedTo": str(filename),
        })

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response.encode())

    def log_message(self, format, *args):
        # Cleaner log output
        sys.stdout.write(f"[{datetime.now().strftime('%H:%M:%S')}] {args[0]}\n")


def main():
    parser = argparse.ArgumentParser(description="Prior Authorization Portal Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve on (default: 8080)")
    args = parser.parse_args()

    server = HTTPServer(("localhost", args.port), PARequestHandler)
    print(f"Prior Authorization Portal running at http://localhost:{args.port}")
    print("Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
