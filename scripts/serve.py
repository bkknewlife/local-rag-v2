#!/usr/bin/env python3
"""Convenience wrapper — can also use ``rag-serve`` after pip install."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rag_eval.cli import cli_serve
cli_serve()
