import subprocess
import unittest
from pathlib import Path


class VllmServerRolloutScriptsTest(unittest.TestCase):
    def test_sft_export_script_copies_standalone_model_and_normalizes_tokenizer_config(self):
        script = Path("scripts/export_sft_model.sh").read_text()
        self.assertIn('EXPORT_DIR', script)
        self.assertIn('extra_special_tokens', script)
        self.assertIn('data["extra_special_tokens"] = {}', script)
        self.assertIn('shutil.copy2(item, target)', script)
        self.assertNotIn('target.symlink_to', script)

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

    def test_h20_lightning_opd_config_matches_long_rollout_experiment(self):
        config = Path("configs/lightning_opd/qwen3-4b-lightning-opd-mvp-h20.py").read_text()
        self.assertIn('"--num-rollout 100 "', config)
        self.assertIn('"--rollout-batch-size 128 "', config)
        self.assertIn('"--rollout-max-response-len 8192 "', config)
        self.assertIn('"--max-tokens-per-gpu 16384 "', config)

        docs = Path("docs/mvp-h20-24h.md").read_text()
        self.assertIn("Qwen3-4B-Base-SFT", docs)
        self.assertIn("--num-samples 12800", docs)
        self.assertIn("--max-tokens 8192", docs)
        self.assertIn("MAX_RESPONSE_LEN=8192", docs)

    def test_shell_scripts_parse_after_server_mode_changes(self):
        for path in ["data_curation/run_curation.sh", "scripts/collect_rollouts.sh"]:
            result = subprocess.run(["bash", "-n", path], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
