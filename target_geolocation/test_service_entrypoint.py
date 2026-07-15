import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from target_geolocation.service_entrypoint import prepare, validate


class ServiceEntrypointTests(unittest.TestCase):
    def test_validate_and_prepare_create_one_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            model = root / "model.engine"
            runs = root / "runs"
            runtime = root / "runtime"
            model.write_bytes(b"test")
            config.write_text(
                json.dumps(
                    {
                        "camera": {
                            "calibrated": True,
                            "camera_id": "test",
                            "image_width": 1280,
                            "image_height": 720,
                            "camera_matrix": [
                                [900.0, 0.0, 640.0],
                                [0.0, 900.0, 360.0],
                                [0.0, 0.0, 1.0],
                            ],
                            "distortion": [0, 0, 0, 0, 0],
                            "rotation_body_from_camera": [
                                [0, -1, 0],
                                [1, 0, 0],
                                [0, 0, 1],
                            ],
                            "lever_arm_body_m": [0, 0, 0],
                        }
                    }
                ),
                encoding="utf-8",
            )
            environment = {
                "GROUND_TARGET_PYTHON": sys.executable,
                "GROUND_TARGET_CONFIG": str(config),
                "GROUND_TARGET_MODEL": str(model),
                "GROUND_TARGET_RUNS_DIR": str(runs),
                "GROUND_TARGET_RUNTIME_DIR": str(runtime),
                "GROUND_TARGET_OUTPUT_WIDTH": "1280",
                "GROUND_TARGET_OUTPUT_HEIGHT": "720",
            }
            with patch.dict(os.environ, environment, clear=False):
                self.assertEqual(validate(), (config, model))
                self.assertEqual(prepare(), 0)

            run_directory = Path(
                (runtime / "run-dir").read_text(encoding="utf-8").strip()
            )
            self.assertTrue((run_directory / "camera").is_dir())
            self.assertEqual((runs / "latest").resolve(), run_directory.resolve())


if __name__ == "__main__":
    unittest.main()
