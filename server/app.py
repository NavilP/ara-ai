"""Root-level wrapper that exposes the real app from my_env.server.app."""

from my_env.server.app import app as app
from my_env.server.app import main as _main


def main() -> None:
    _main()


__all__ = ["app", "main"]


if __name__ == "__main__":
    main()
