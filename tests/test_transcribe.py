import os
import pytest
from unittest.mock import Mock, patch
from CallChain.audio.transcribe import AudioTranscriber

class TestAudioTranscriber:
    @pytest.fixture
    def mock_client(self):
        """Create a mock Groq client."""
        with patch('groq.Groq') as mock_groq:
            mock_client = Mock()
            mock_groq.return_value = mock_client
            yield mock_client

    def test_transcribe_success(self, mock_client):
        """Test successful transcription."""
        # Setup
        mock_audio = Mock()
        mock_audio.audio.transcriptions.create.return_value = "Test transcription"
        mock_client.return_value = mock_audio
        
        transcriber = AudioTranscriber(api_key="test_api_key")
        
        # Execute
        result = transcriber.transcribe("test_audio.mp3")
        
        # Assert
        assert result == "Test transcription"
        mock_audio.audio.transcriptions.create.assert_called_once()

    def test_transcribe_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent files."""
        transcriber = AudioTranscriber(api_key="test_api_key")
        
        with pytest.raises(FileNotFoundError):
            transcriber.transcribe("non_existent_file.mp3")

    @patch('os.getenv', return_value=None)
    def test_init_no_api_key(self, mock_getenv):
        """Test that ValueError is raised when no API key is provided."""
        with pytest.raises(ValueError) as excinfo:
            AudioTranscriber()
        assert "No API key provided" in str(excinfo.value)

    def test_transcribe_api_error(self, mock_client):
        """Test handling of API errors during transcription."""
        mock_audio = Mock()
        mock_audio.audio.transcriptions.create.side_effect = Exception("API Error")
        mock_client.return_value = mock_audio
        
        transcriber = AudioTranscriber(api_key="test_api_key")
        
        with pytest.raises(Exception) as excinfo:
            transcriber.transcribe("test_audio.mp3")
        assert "Error during transcription" in str(excinfo.value)