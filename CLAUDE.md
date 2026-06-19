# CLAUDE.md — Project Conventions

## Environment

Use the `smartrobox` conda environment for all development and testing:

```bash
conda activate smartrobox
```

Python version: **3.11**

---

## Python Coding Style

All Python files in this project follow the style established in `server/app/plc_app.py`.

---

### File Header

Every Python file must start with a Doxygen block header:

```python
##
# @file filename.py
#
# @brief One-line description of what this file does.
#
# @section author Author(s)
# - Created by XXXX on YYYY/MM/DD.
#
# Copyright (c) 2026 HACHIX.  All rights reserved.
```

---

### Import Sections

Group imports with labeled comments. Order: Standard → External → Internal.

```python
# Standard library
import os
import threading

# External library
import pytz

# Internal library
from config.config import ConfigInstance
from domain.image_result import ImageResult
```

---

### Class Docstrings

Use `"""! description."""` (Doxygen `!` prefix).

```python
class FeatureTracker:
    """! Collects ImageResults per feature and signals when a feature is complete.

    Wraps ProductTracker to provide a focused, feature-level API for the
    detection pipeline. Thread-safe.
    """
```

---

### Method Docstrings

Use `@param name<type>:` and `@return<type>:` Doxygen tags. Include a leading `"""!`.

```python
def add_capture(self, image_result: ImageResult) -> FeatureResult | None:
    """! Register a capture result and return FeatureResult if feature is complete.

    @param image_result<ImageResult>: Completed capture to register.
    @return<FeatureResult|None>: Completed FeatureResult, or None if the
        feature still expects more captures.
    """
```

For methods that follow the existing `config.py` pattern (Note / Returns prose style), keep consistency within that file.

---

### Section Separators Inside Classes

Use `# ===` separators to divide public and private sections:

```python
class MyClass:

    # =========================================================================
    # PUBLIC METHODS
    # =========================================================================

    def public_method(self):
        ...

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _private_method(self):
        ...
```

---

### Private Members

- Private instance attributes: leading underscore (`self._config`, `self._lock`)
- Private class constants: leading underscore (`_PLC_OK = 1`, `_SERVER_DOWN = 0`)
- Private methods: leading underscore (`_build_expected_captures`, `_reconnect`)
- Public constants stay without underscore only when they are part of the public API

---

### General Rules

- **No trailing summaries** in responses — the diff speaks for itself.
- **No extra abstractions** beyond what the task requires.
- **No comments** on self-evident code; comments only where logic is non-obvious.
- **Type hints** are used on method signatures (`str`, `dict`, `list[str]`, `X | None`).
- **dataclasses** are used for pure data containers (domain objects).
- **Thread safety**: shared service instances use `threading.Lock`.
- **Singleton pattern**: use `SingletonClass` metaclass (see `config/config.py`) for services that must have one instance per process.
