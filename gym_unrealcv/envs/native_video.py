"""Control UE-native VAT video recording without per-frame network commands."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path


class NativeVATVideoController:
    ENV_NAME = "UNREALCV_VAT_RECORDING_CONTROL"

    def __init__(self, control_path: Path, timeout: float = 30.0):
        self.control_path = Path(control_path).resolve()
        self.status_path = Path(f"{self.control_path}.status.json")
        self.timeout = timeout
        self.started = False

    def configure_child_process(self) -> None:
        self.control_path.parent.mkdir(parents=True, exist_ok=True)
        for path in (self.control_path, self.status_path):
            if path.exists():
                path.unlink()
        os.environ[self.ENV_NAME] = str(self.control_path)

    def start(self, camera_id: int, output_directory: Path, fps: int = 30) -> dict:
        request_id = uuid.uuid4().hex
        self._write_request({
            "request_id": request_id,
            "command": "start",
            "camera_id": int(camera_id),
            "output_directory": str(Path(output_directory).resolve()),
            "fps": int(fps),
        })
        status = self._wait_for(request_id, {"recording", "error"})
        if status["state"] != "recording":
            raise RuntimeError(f"Native VAT video start failed: {status}")
        self.started = True
        return status

    def stop(self) -> dict:
        if not self.started:
            raise RuntimeError("Native VAT video was not started")
        request_id = uuid.uuid4().hex
        self._write_request({"request_id": request_id, "command": "stop"})
        status = self._wait_for(request_id, {"completed", "error"})
        self.started = False
        if status["state"] != "completed":
            raise RuntimeError(f"Native VAT video stop failed: {status}")
        return status

    def _write_request(self, request: dict) -> None:
        temporary = self.control_path.with_suffix(self.control_path.suffix + ".tmp")
        temporary.write_text(json.dumps(request), encoding="utf-8")
        os.replace(temporary, self.control_path)

    def _wait_for(self, request_id: str, terminal_states: set[str]) -> dict:
        deadline = time.monotonic() + self.timeout
        last_status = None
        while time.monotonic() < deadline:
            if self.status_path.exists():
                try:
                    status = json.loads(self.status_path.read_text(encoding="utf-8-sig"))
                except (json.JSONDecodeError, OSError):
                    time.sleep(0.05)
                    continue
                last_status = status
                if status.get("request_id") == request_id and status.get("state") in terminal_states:
                    return status
            time.sleep(0.05)
        raise TimeoutError(f"Timed out waiting for native VAT video status; last={last_status}")

