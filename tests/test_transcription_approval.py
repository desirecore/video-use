import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


HELPERS = Path(__file__).resolve().parents[1] / "helpers"
sys.path.insert(0, str(HELPERS))
import transcribe
import transcribe_batch


class TranscriptionApprovalTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.videos = Path(temp.name)
        for name in ["approved.mp4", "excluded.mp4"]:
            subprocess.run(
                ["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                 "sine=frequency=440:duration=0.25", str(self.videos / name)],
                check=True, capture_output=True,
            )
        post = patch.object(transcribe.requests, "post")
        self.post = post.start()
        self.addCleanup(post.stop)
        self.post.return_value = Mock(status_code=200)
        self.post.return_value.json.return_value = {"words": []}
        environ = patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test-environment-key"})
        environ.start()
        self.addCleanup(environ.stop)

    def run_cli(self, module, path, *args):
        output = io.StringIO()
        with patch.object(sys, "argv", [module.__file__, str(path), *args]):
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                module.main()
        return output.getvalue()

    def test_inventory_lists_uncached_media_without_a_request(self):
        output = self.run_cli(transcribe_batch, self.videos)
        self.assertIn("ElevenLabs Scribe", output)
        self.assertIn("approved.mp4", output)
        self.assertIn("excluded.mp4", output)
        self.post.assert_not_called()
        self.assertFalse((self.videos / "edit").exists())

    def test_batch_approval_requires_named_files(self):
        with self.assertRaises(SystemExit) as error:
            self.run_cli(transcribe_batch, self.videos, "--upload-approved")
        self.assertEqual(error.exception.code, 2)
        self.post.assert_not_called()

    def test_approved_selection_uses_environment_key_and_reuses_cache(self):
        self.run_cli(
            transcribe_batch, self.videos, "--file=approved.mp4", "--upload-approved"
        )
        self.post.assert_called_once()
        url, = self.post.call_args.args
        sent = self.post.call_args.kwargs
        self.assertEqual(url, "https://api.elevenlabs.io/v1/speech-to-text")
        self.assertEqual(sent["files"]["file"][0], "approved.wav")
        self.assertEqual(sent["headers"]["xi-api-key"], "test-environment-key")
        self.assertEqual(sent["data"]["model_id"], "scribe_v2")
        self.assertFalse(sent["allow_redirects"])
        self.assertTrue((self.videos / "edit" / "transcripts" / "approved.json").exists())
        self.assertFalse((self.videos / "edit" / "transcripts" / "excluded.json").exists())
        self.post.reset_mock()
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": ""}):
            self.assertIn("cached:", self.run_cli(transcribe, self.videos / "approved.mp4"))
        self.post.assert_not_called()

    def test_single_uncached_file_requires_approval(self):
        with self.assertRaises(SystemExit) as error:
            self.run_cli(transcribe, self.videos / "approved.mp4")
        self.assertEqual(error.exception.code, 2)
        self.post.assert_not_called()

    def test_candidate_rejects_missing_key_before_request(self):
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": ""}):
            with self.assertRaisesRegex(SystemExit, "runtime environment"):
                self.run_cli(transcribe, self.videos / "approved.mp4", "--upload-approved")
        self.post.assert_not_called()
