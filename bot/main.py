"""Entry point: python3 -u bot/main.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mf  # noqa: E402
from brain import Brain  # noqa: E402

if __name__ == "__main__":
    mf.run(Brain)
