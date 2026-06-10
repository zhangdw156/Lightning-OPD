import subprocess
import unittest
from pathlib import Path


class VllmServerRolloutScriptsTest(unittest.TestCase):
    def test_run_curation_uses_vllm_serve_when_server_mode_enabled(self):
        script = Path("data_curation/run_curation.sh").read_text()
        self.assertIn('source "${VLLM_VENV}/bin/activate"', script)
        self.assertIn('vllm serve "${MODEL_ARG}"', script)
        self.assertIn('--served-model-name "${SERVED_MODEL_NAME}"', script)
        self.assertIn('--server-url "http://${VLLM_HOST}:${PORT}/v1"', script)
        self.assertIn('${SCRIPT_DIR}/pipeline.py', script)

    def test_pipeline_supports_openai_compatible_server_mode(self):
        pipeline = Path("data_curation/pipeline.py").read_text()
        self.assertIn("--server-url", pipeline)
        self.assertIn("chat/completions", pipeline)
        self.assertIn("run_server_batch", pipeline)
        self.assertIn("completion_tokens", pipeline)
        self.assertIn("urllib.request.urlopen", pipeline)

    def test_shell_scripts_parse_after_server_mode_changes(self):
        for path in ["data_curation/run_curation.sh", "scripts/collect_rollouts.sh"]:
            result = subprocess.run(["bash", "-n", path], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
