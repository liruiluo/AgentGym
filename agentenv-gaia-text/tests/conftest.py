from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

# Importing agentenv.controller normally loads the model/training stack.  These
# adapter tests need only its dependency-free env/types modules.
if "agentenv.controller" not in sys.modules:
    controller = ModuleType("agentenv.controller")
    controller.__path__ = [
        str(
            Path(__file__).resolve().parents[2] / "agentenv" / "agentenv" / "controller"
        )
    ]
    sys.modules["agentenv.controller"] = controller

if "agentenv.envs" not in sys.modules:
    envs = ModuleType("agentenv.envs")
    envs.__path__ = [
        str(Path(__file__).resolve().parents[2] / "agentenv" / "agentenv" / "envs")
    ]
    sys.modules["agentenv.envs"] = envs
