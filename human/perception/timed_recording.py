"""Record camera frames off-thread, preserving their host-clock timestamps."""
from __future__ import annotations

import csv
import json
import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import queue
import threading
import time

import cv2


def clock_identity() -> dict:
    boot = Path('/proc/sys/kernel/random/boot_id')
    return {'clock': 'time.monotonic_ns',
            'boot_id': boot.read_text().strip() if boot.exists() else None}


def capture_source_provenance(directory, root):
    """Freeze loaded project Python modules, including local source edits."""
    root, directory = Path(root).resolve(), Path(directory)
    hashes, unavailable = {}, []
    for module_name, module in list(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix != '.py' or not path.is_relative_to(root):
            continue
        relative = path.relative_to(root)
        if any(part in ('.venv', '.git') for part in relative.parts):
            continue
        try:
            raw = path.read_bytes()
        except (FileNotFoundError, IsADirectoryError) as exc:
            # torch.ops/classes advertise synthetic relative __file__ values.
            # They resolve under cwd but are not source files on disk. Keep
            # omissions explicit, including real modules deleted since import.
            unavailable.append({'module': module_name, 'path': str(relative),
                                'reason': type(exc).__name__})
            continue
        destination = directory / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        hashes[str(relative)] = hashlib.sha256(raw).hexdigest()
    def git(*args):
        result = subprocess.run(['git', *args], cwd=root, capture_output=True,
                                text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    return {'commit': git('rev-parse', 'HEAD'),
            'working_tree_status': git('status', '--porcelain', '--untracked-files=normal'),
            'source_sha256': hashes,
            'unavailable_sources': unavailable,
            'source_scope': 'Loaded project Python modules at camera recorder initialization'}


class TimedCameraRecorder:
    """AVI frames + CSV acquisition times; final replay supplies real timing.

    The AVI's nominal 30 fps is only an indexed frame container. Capture never
    waits for encoding; overload drops frames explicitly, retaining timestamps
    so offline exports hold the previous frame instead of speeding up time.
    """

    def __init__(self, directory, metadata=None, queue_size=60):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.metadata = {**clock_identity(), **(metadata or {}),
                         'created_at_utc': datetime.now(timezone.utc).isoformat(),
                         'complete': False, 'nominal_fps': 30,
                         'timing': 'Use frames.csv, not AVI playback rate',
                         'mirrored': False, 'overlay': False}
        source_root = self.metadata.pop('source_root', None)
        if source_root:
            self.metadata['provenance'] = capture_source_provenance(self.directory, source_root)
        self._queue = queue.Queue(maxsize=queue_size)
        self._error = None
        self._closed = False
        self.dropped = 0
        self.frames = 0
        self._write_metadata()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _write_metadata(self):
        temporary = self.directory / 'metadata.tmp.json'
        temporary.write_text(json.dumps(self.metadata, indent=2) + '\n')
        temporary.replace(self.directory / 'metadata.json')

    def submit(self, frame, timestamp_ns=None):
        if self._error is not None:
            raise RuntimeError('Camera recording failed') from self._error
        if self._closed:
            raise RuntimeError('Camera recording is closed')
        timestamp_ns = time.monotonic_ns() if timestamp_ns is None else timestamp_ns
        try:
            self._queue.put_nowait((int(timestamp_ns), frame.copy()))
        except queue.Full:
            self.dropped += 1
            if self.dropped == 1:
                print('[camera recording] Encoder overloaded; dropped frames will be reported.')

    def _worker(self):
        writer = None
        try:
            with (self.directory / 'frames.csv').open('w', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(['frame', 'monotonic_ns'])
                while True:
                    item = self._queue.get()
                    if item is None:
                        break
                    timestamp_ns, frame = item
                    if writer is None:
                        h, w = frame.shape[:2]
                        writer = cv2.VideoWriter(str(self.directory / 'camera.avi'),
                                                 cv2.VideoWriter_fourcc(*'MJPG'), 30, (w, h))
                        if not writer.isOpened():
                            raise RuntimeError('Cannot open MJPEG camera recording')
                        self.metadata.update(width=w, height=h, start_ns=timestamp_ns)
                    if frame.shape[:2] != (self.metadata['height'], self.metadata['width']):
                        raise ValueError('Camera resolution changed during recording')
                    writer.write(frame)
                    csv_writer.writerow([self.frames, timestamp_ns])
                    f.flush()
                    self.frames += 1
                    self.metadata['end_ns'] = timestamp_ns
        except BaseException as exc:
            self._error = exc
        finally:
            if writer is not None:
                writer.release()

    def close(self):
        if self._closed:
            return
        self._closed = True
        while self._thread.is_alive():
            try:
                self._queue.put(None, timeout=0.1)
                break
            except queue.Full:
                continue
        self._thread.join()
        self.metadata.update(complete=self._error is None, frames=self.frames,
                             dropped_frames=self.dropped,
                             error=str(self._error) if self._error else None)
        self._write_metadata()
        if self._error is not None:
            raise RuntimeError('Camera recording failed') from self._error
        print(f'[camera recording] {self.frames} frames, {self.dropped} dropped -> {self.directory}')
