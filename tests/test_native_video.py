import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

from gym_unrealcv.envs.native_video import NativeVATVideoController


class NativeVATVideoControllerTest(unittest.TestCase):
    def test_start_and_stop_handshake(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            control = NativeVATVideoController(Path(temp_dir) / "control.json", timeout=2.0)
            control.configure_child_process()
            self.assertEqual(os.environ[control.ENV_NAME], str(control.control_path))

            def server():
                handled = set()
                while len(handled) < 2:
                    if not control.control_path.exists():
                        time.sleep(0.01)
                        continue
                    request = json.loads(control.control_path.read_text(encoding="utf-8"))
                    request_id = request["request_id"]
                    if request_id in handled:
                        time.sleep(0.01)
                        continue
                    handled.add(request_id)
                    state = "recording" if request["command"] == "start" else "completed"
                    control.status_path.write_text(
                        json.dumps({"request_id": request_id, "state": state}), encoding="utf-8"
                    )

            thread = threading.Thread(target=server)
            thread.start()
            started = control.start(2, Path(temp_dir) / "video", fps=24)
            stopped = control.stop()
            thread.join(timeout=2.0)

            self.assertEqual(started["state"], "recording")
            self.assertEqual(stopped["state"], "completed")
            self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
