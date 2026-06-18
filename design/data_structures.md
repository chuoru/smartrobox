# Data Structures

## Device Interface

All hardware devices follow the same lifecycle contract. Orbbec uses `start()`/`stop()` instead of `open()`/`close()`, but Controller normalises this internally.

| Field / Method | Type | Role |
|----------------|------|------|
| `_debug` | `bool` | Skip real hardware; return mock responses |
| `open()` | `→ bool` | Connect to hardware; return False on failure |
| `close()` | `→ None` | Disconnect and release resources |
| `is_opened()` | `→ bool` | True if connection is active |

### FairinoInterface (robot arm)

| Field | Type | Role |
|-------|------|------|
| `_robot` | `Robot` (ctypes binding) | Fairino DLL handle |
| `_is_opened` | `bool` | Connection state |

| Method | Signature | Returns |
|--------|-----------|---------|
| `movej` | `(j1..j6: float, vel: float)` | `bool` — success |
| `movel` | `(x,y,z,rx,ry,rz: float, vel: float)` | `bool` |
| `move` | `(x,y,z,rx,ry,rz: float)` | `bool` |
| `tpos` | `()` | `(error_code: int, [x,y,z,rx,ry,rz])` |
| `get_inverse_kinematics` | `(x,y,z,rx,ry,rz: float)` | `(error_code, [j1..j6])` |

### LinkerbotInterface (robotic hand)

| Field | Type | Role |
|-------|------|------|
| `_api` | `LinkerHandApi` | CAN/RS485 driver wrapper |
| `_is_opened` | `bool` | Connection state |

| Method | Signature | Returns |
|--------|-----------|---------|
| `move` | `(pose: list[int])` — 0–255 per joint | `bool` |
| `set_speed` | `(speed: list[int])` | `bool` |
| `set_torque` | `(torque: list[int])` | `bool` |
| `get_state` | `()` | `list[int]` — current joint positions |
| `get_touch` | `()` | `list[int]` — 6 touch sensor values |
| `get_force` | `()` | `list[list]` — `[[norm],[tang],[dir],[app]]` |
| `get_fault` | `()` | `list` — fault codes |
| `clear_faults` | `()` | `bool` |

### OrbbecInterface (depth camera)

| Field | Type | Role |
|-------|------|------|
| `_thread` | `threading.Thread` | Background capture loop |
| `_running` | `bool` | Loop control flag |
| `_lock` | `threading.Lock` | Guards frame buffers |
| `_color_frame` | `ndarray \| None` | Latest BGR frame (H×W×3, uint8) |
| `_depth_frame` | `ndarray \| None` | Latest depth map (H×W, uint16) |
| `_fx, _fy, _cx, _cy` | `float` | Camera intrinsics |
| `_depth_scale` | `float` | Depth → metres factor |

| Method | Signature | Returns |
|--------|-----------|---------|
| `start` | `()` | `bool` |
| `stop` | `()` | `None` |
| `is_alive` | `()` | `bool` |
| `get_color_frame` | `()` | `ndarray \| None` |
| `get_depth_frame` | `()` | `ndarray \| None` |
| `pixel_to_world` | `(u, v: int)` | `(X, Y, Z): tuple[float,float,float]` |

---

## Controller

Central device registry. Builds from `device.yaml` at startup.

### `_DeviceEntry` (dataclass)

| Field | Type | Role |
|-------|------|------|
| `name` | `str` | Unique key in registry (e.g. `"left_arm"`) |
| `type` | `str` | Device class key (e.g. `"fairino"`) |
| `instance` | `object` | Live device interface instance |

### `Controller`

| Field | Type | Role |
|-------|------|------|
| `_registry` | `dict[str, _DeviceEntry]` | Name → entry map |
| `_lock` | `threading.Lock` | Guards registry access |
| `_DEVICE_FACTORIES` | `dict[str, type]` | Type string → constructor (module-level constant) |

| Method | Signature | Returns |
|--------|-----------|---------|
| `open` | `(device_name: str)` | `bool` |
| `close` | `(device_name: str)` | `bool` |
| `open_all` | `()` | `dict[str, bool]` |
| `close_all` | `()` | `dict[str, bool]` |
| `list_devices` | `()` | `list[str]` |
| `status` | `(device_name: str)` | `dict` — `{name, type, is_opened}` |
| `execute` | `(device_name, method, *args, **kwargs)` | `object` — device return value |

---

## BaseAction

Abstract base for long-running multi-device operations. Subclasses override `_run()`.

### `ActionState` (Enum)

| Value | Meaning |
|-------|---------|
| `IDLE` | Not yet started |
| `RUNNING` | Thread active, executing steps |
| `PAUSED` | Thread blocked at next checkpoint |
| `DONE` | `_run()` returned normally and was not cancelled |
| `FAILED` | `_run()` raised an unhandled exception |
| `CANCELLED` | `cancel()` was requested and `_run()` exited cleanly |

### `BaseAction` fields

| Field | Type | Role |
|-------|------|------|
| `_controller` | `Controller` | Device dispatch target |
| `_state` | `ActionState` | Current lifecycle state (guarded by `_lock`) |
| `_lock` | `threading.Lock` | Guards all state reads/writes |
| `_pause_event` | `threading.Event` | Set = running; Clear = paused. Used by `_checkpoint()` |
| `_done_event` | `threading.Event` | Set on any terminal state; used by `wait()` |
| `_cancelled` | `bool` | Set by `cancel()`; read inside `_checkpoint()` |
| `_thread` | `Thread \| None` | Background worker thread (daemon=True) |
| `_result` | `object` | Return value of `_run()` on DONE; safe to read after `wait()` |
| `_error` | `Exception \| None` | Exception captured on FAILED; safe to read after `wait()` |

### State Transition Authority

`_thread_entry` is the **sole writer** of terminal states (DONE, FAILED, CANCELLED). `cancel()` only sets the flag and unblocks the checkpoint — it never writes `_state` directly. This prevents the race where `cancel()` and `_thread_entry` both try to write the final state.

```
cancel() called              _thread_entry (in background thread)
    │                               │
    ├─ _cancelled = True            └─ [after _run() returns]
    └─ _pause_event.set()               with _lock:
         (unblocks checkpoint)              _state = CANCELLED  ← sole writer
                                        _done_event.set()
```

### `_checkpoint()` contract

Subclasses must call `_checkpoint()` after each logical step. If it returns `False`, they must return immediately from `_run()`.

```python
_pause_event.wait()    # blocks when paused; unblocks on resume() or cancel()
return not _cancelled  # False → return from _run() immediately
```

### Subclass pattern

```python
class PickAction(BaseAction):
    def _run(self):
        self._call("arm", "movel", x, y, z + 50, rx, ry, rz)
        if not self._checkpoint(): return

        self._call("arm", "movel", x, y, z, rx, ry, rz)
        if not self._checkpoint(): return

        self._call("hand", "move", close_pose)
        if not self._checkpoint(): return

        self._call("arm", "movel", x, y, z + 100, rx, ry, rz)
```

### Composition patterns

```python
# Consecutive
a1.start(); a1.wait(); a2.start(); a2.wait()

# Parallel
a1.start(); a2.start(); a1.wait(); a2.wait()

# Check outcome before chaining
if a1.start() and a1.wait():   # True only if DONE
    a2.start()
```
