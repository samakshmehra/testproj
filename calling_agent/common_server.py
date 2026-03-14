"""Entry point for the common calling server."""

import sys
from pathlib import Path

from flask import Flask

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calling_agent.services.routes import register_routes, run_server

app = Flask(__name__)
register_routes(app)


def main() -> None:
    run_server(app)


if __name__ == "__main__":
    main()
