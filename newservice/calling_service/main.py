from __future__ import annotations

import uvicorn

from .app import app
from .runtime import runtime


def main() -> None:
    uvicorn.run(
        app,
        host=runtime.settings.host,
        port=runtime.settings.port,
    )


if __name__ == "__main__":
    main()
