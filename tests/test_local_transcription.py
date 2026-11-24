import importlib.util
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "helpers"))


@unittest.skipUnless(
    importlib.util.find_spec("speechbrain"), "requires the local extra"
)
class CalibrationTests(unittest.TestCase):
    def test_keeps_words_the_acoustic_model_cannot_normalize(self):
        from local_asr.pipeline import Calibration

        calibration = Calibration.__new__(Calibration)
        calibration.vocab = {"h": 0}
        words = [{"id": 187, "text": "-huh.", "start": 1, "end": 1.2}]
        self.assertEqual(
            calibration.word_evidence(words, {"mix": object()}), {"187": {}}
        )
        self.assertEqual(words[0]["text"], "-huh.")
