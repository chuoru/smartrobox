import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(__file__))

# Stub hardware SDKs that are not installed in the test environment.
# Device classes import these at module level; without stubs the entire
# controller module fails to load and no unit tests can run.
_pyorbbecsdk_mock = MagicMock()
_pyorbbecsdk_mock.OBFormat.MJPG = "MJPG"
_pyorbbecsdk_mock.OBFormat.RGB = "RGB"
_pyorbbecsdk_mock.OBFormat.BGR = "BGR"
sys.modules.setdefault("pyorbbecsdk", _pyorbbecsdk_mock)

# Stub ultralytics so action modules that import YOLO at module level can be
# collected without the package installed in the test environment.
sys.modules.setdefault("ultralytics", MagicMock())

# Stub mediapipe so estimate_hand imports cleanly without the package installed.
sys.modules.setdefault("mediapipe", MagicMock())
