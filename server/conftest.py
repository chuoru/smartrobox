import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(__file__))

# Stub hardware SDKs that are not installed in the test environment.
# Device classes import these at module level; without stubs the entire
# controller module fails to load and no unit tests can run.
sys.modules.setdefault("pyorbbecsdk", MagicMock())
