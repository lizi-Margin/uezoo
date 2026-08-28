import os
import tempfile
import unittest
from pathlib import Path

from gym_unrealcv.envs.base_env import _ensure_external_binary_unrealcv_ini


class ExternalBinaryIniTest(unittest.TestCase):
    def test_creates_launcher_compatible_ini_next_to_binary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            binary = Path(temp_dir) / 'ExampleProject.exe'
            binary.touch()

            ini_path = Path(_ensure_external_binary_unrealcv_ini(os.fspath(binary)))

            self.assertEqual(ini_path, binary.with_name('unrealcv.ini'))
            self.assertEqual(
                ini_path.read_text(encoding='utf-8'),
                '[UnrealCV.Core]\nPort=9000\nWidth=640\nHeight=480\n',
            )

    def test_preserves_existing_ini(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            binary = Path(temp_dir) / 'ExampleProject.exe'
            binary.touch()
            ini_path = binary.with_name('unrealcv.ini')
            ini_path.write_text('custom configuration', encoding='utf-8')

            _ensure_external_binary_unrealcv_ini(os.fspath(binary))

            self.assertEqual(ini_path.read_text(encoding='utf-8'), 'custom configuration')


if __name__ == '__main__':
    unittest.main()
