"""Точка входу: python -m satimg <команда> [параметри]."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
