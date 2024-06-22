from pprint import pprint
import whisper
from pathlib import Path

AUDIO_DIR = Path(__file__).parent / "test_audio_files"
model = whisper.load_model("medium") # model size: base / small / medium / large


def detect_language_and_transcribe(audio_file: str):
    audio = whisper.load_audio(audio_file)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, language_probs = model.detect_language(mel)
    language: str = max(language_probs, key=language_probs.get)
    pprint(f"Detected language: {language}")
    options = whisper.DecodingOptions(language=language, task="transcribe")
    result = whisper.decode(model, mel, options)
    pprint(result)
    return result.text


## TRANSCRIBE FILE
dutch_test = detect_language_and_transcribe(
    str(AUDIO_DIR / "dutch_the_netherlands.mp3")
)
result = model.transcribe(str(AUDIO_DIR / "dutch_the_netherlands.mp3"), verbose=True)
pprint(result["text"])


## TRANSCRIBE LONG FILE
# result = model.transcribe(
#     str(AUDIO_DIR / "dutch_long_repeat_file.mp3"),
#     verbose=True,
#     language="nl",
#     task="transcribe",
# )
# pprint(result["text"])


## TRANSLATE
# result = model.transcribe(
#     str(AUDIO_DIR / "dutch_the_netherlands.mp3"),
#     verbose=True,
#     language="nl",
#     task="translate",
# )
# pprint(result["text"])
