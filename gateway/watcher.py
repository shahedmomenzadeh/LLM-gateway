"""
Hot-reload watcher for config.yaml.

Uses watchdog to observe the config file. On any change, attempts to load the
new config; if validation succeeds, atomically swaps it into the runtime
State (preserving cooldowns). If validation fails, logs the error and keeps
the old config running.

Runs in a background thread (watchdog's API is synchronous). Started from
main.py's lifespan.
"""
from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Optional

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from .config import load_config_or_none
from .logging_setup import get_logger
from .state import State

log = get_logger("gateway.watcher")


class _ConfigChangeHandler(FileSystemEventHandler):
    def __init__(self, config_path: Path, state: State, loop: asyncio.AbstractEventLoop):
        self.config_path = config_path.resolve()
        self.state = state
        self.loop = loop
        # Debounce: watchdog fires multiple events per save
        self._last_mtime: float = 0.0

    def on_modified(self, event):  # noqa: N802 - watchdog API
        if event.is_directory:
            return
        try:
            p = Path(event.src_path).resolve()
        except Exception:
            return
        if p != self.config_path:
            return
        # Debounce via mtime
        try:
            mtime = p.stat().st_mtime
        except OSError:
            return
        if mtime == self._last_mtime:
            return
        self._last_mtime = mtime
        # Schedule the reload on the asyncio loop
        self.loop.call_soon_threadsafe(
            asyncio.ensure_future, self._do_reload()
        )

    async def _do_reload(self) -> None:
        log.info("config file changed — attempting reload")
        new_cfg = load_config_or_none(self.config_path)
        if new_cfg is None:
            log.error("config reload failed — keeping previous config")
            return
        # Swap atomically
        self.state.reload(new_cfg)
        log.info(
            "config reloaded successfully",
            extra={
                "providers": list(new_cfg.providers.keys()),
                "waterfall_steps": len(new_cfg.waterfall),
            },
        )


class ConfigWatcher:
    """Wraps watchdog observer + handler. Call start() / stop()."""

    def __init__(self, config_path: Path, state: State, loop: asyncio.AbstractEventLoop):
        self.config_path = config_path
        self.state = state
        self.loop = loop
        self._observer: Optional[Observer] = None
        self._handler: Optional[_ConfigChangeHandler] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not self.config_path.exists():
            log.warning(f"config file {self.config_path} does not exist; watcher disabled")
            return
        self._handler = _ConfigChangeHandler(self.config_path, self.state, self.loop)
        self._observer = Observer()
        # Watch the parent directory (watching the file itself is unreliable
        # across editors that write-then-rename)
        watch_dir = str(self.config_path.resolve().parent)
        self._observer.schedule(self._handler, watch_dir, recursive=False)
        self._observer.daemon = True
        self._observer.start()
        log.info(f"watching {self.config_path.resolve()} for changes")

    def stop(self) -> None:
        if self._observer:
            try:
                self._observer.stop()
                self._observer.join(timeout=2.0)
            except Exception:
                pass
        self._observer = None
        self._handler = None
