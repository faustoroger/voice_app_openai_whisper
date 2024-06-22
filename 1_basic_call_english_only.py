import whisper
from pathlib import Path
from pprint import pprint

MODEL = whisper.load_model("small.en")
AUDIO_DIR = Path(__file__).parent / "test_audio_files"

def get_transcription(audio_file: str):
    result = MODEL.transcribe(audio_file)
    pprint(result)
    return result

get_transcription(str(AUDIO_DIR / "terrible_quality.mp3"))
