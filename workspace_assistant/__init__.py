"""Lets `adk web` (run from the repository root) load this folder as an agent.

ADK looks for `root_agent`. The project uses absolute imports (config, tools),
so this folder is added to sys.path first.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import create_agent  # noqa: E402

root_agent = create_agent()
