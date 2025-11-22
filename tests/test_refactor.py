import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Add the project root to the path so we can import CallChain
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from CallChain.audio.config import AudioConfig
from CallChain.audio.processor import AudioProcessor
from CallChain.audio.transcribe import AudioTranscriber

class TestAudioRefactor(unittest.TestCase):
    def setUp(self):
        self.api_key = "test_key"
        self.config = AudioConfig(api_key=self.api_key)

    def test_config_initialization(self):
        """Test that AudioConfig initializes correctly."""
        config = AudioConfig(api_key="test", model="whisper-v3")
        self.assertEqual(config.api_key, "test")
        self.assertEqual(config.model, "whisper-v3")
        self.assertEqual(config.target_sr, 16000)  # Default

    def test_processor_initialization(self):
        """Test that AudioProcessor initializes with config."""
        processor = AudioProcessor(self.config)
        self.assertEqual(processor.config, self.config)

    @patch('CallChain.audio.processor.librosa.load')
    def test_processor_load_audio(self, mock_load):
        """Test audio loading."""
        mock_load.return_value = (np.zeros(100), 16000)
        processor = AudioProcessor(self.config)
        y, sr = processor._load_audio("dummy.mp3")
        self.assertEqual(sr, 16000)
        self.assertEqual(len(y), 100)

    def test_transcriber_initialization_backward_compat(self):
        """Test that AudioTranscriber works with old parameters."""
        with patch.dict(os.environ, {"GROQ_API_KEY": "env_key"}):
            transcriber = AudioTranscriber(noise_reduction=True)
            self.assertTrue(transcriber.config.noise_reduction)
            self.assertEqual(transcriber.config.api_key, "env_key")

    def test_transcriber_initialization_new_config(self):
        """Test that AudioTranscriber works with AudioConfig."""
        config = AudioConfig(api_key="new_key", normalize=False)
        transcriber = AudioTranscriber(config=config)
        self.assertEqual(transcriber.config.api_key, "new_key")
        self.assertFalse(transcriber.config.normalize)

if __name__ == '__main__':
    unittest.main()
