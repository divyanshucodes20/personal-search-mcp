"""Re-index files as they change (debounced). Requires `watchdog`."""
from __future__ import annotations

import threading
import time
from pathlib import Path

from .config import Config
from .indexer import Indexer, is_indexable, log


def watch(cfg: Config, debounce: float = 2.0) -> None:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    pending: dict[str, float] = {}
    lock = threading.Lock()

    class Handler(FileSystemEventHandler):
        def on_any_event(self, event):
            if event.is_directory:
                return
            with lock:
                for p in (getattr(event, "src_path", ""), getattr(event, "dest_path", "")):
                    if p:
                        pending[str(p)] = time.time()

    indexer = Indexer(cfg)
    observer = Observer()
    for root in cfg.roots:
        if Path(root).exists():
            observer.schedule(Handler(), root, recursive=True)
            log(f"[watch] {root}")
    observer.start()
    try:
        while True:  # all SQLite work happens on this thread
            time.sleep(1.0)
            now = time.time()
            with lock:
                ready = [p for p, t in pending.items() if now - t >= debounce]
                for p in ready:
                    pending.pop(p)
            for p in ready:
                path = Path(p)
                if path.exists():
                    if is_indexable(path, cfg):
                        status = indexer.index_file(path)
                        if status in ("new", "updated"):
                            log(f"[{status}] {path}")
                else:
                    if indexer.remove_path(path):
                        log(f"[removed] {path}")
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join()
