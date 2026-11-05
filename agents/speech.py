"""Purpose of this program: turn a short voice answer into text, for the
optional voice input on the New Intake screen.

Uses OpenAI's speech-to-text (gpt-4o-mini-transcribe) with the kit's
existing OPENAI_API_KEY. The text is NOT saved here: the app puts it into
the answer box so the caregiver can check and fix it, and only when they tap
Next does it go through the normal intake path (PII masking first, then the
IntakeAgent).

Privacy: the recording itself goes to OpenAI before any masking can happen,
so the app asks the caregiver not to say the patient's name, phone number or
address out loud.

Try it:  python -m agents.speech path/to/recording.wav
"""

import io
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
MODEL = "gpt-4o-mini-transcribe"
_client = None


def transcribe(audio: bytes, filename: str = "answer.wav") -> str:
    """Return the words spoken in the recording ("" if nothing was heard)."""
    global _client
    _client = _client or OpenAI()  # created on first use, reads OPENAI_API_KEY
    file = io.BytesIO(audio)
    file.name = filename  # the API uses the name to know the audio format
    result = _client.audio.transcriptions.create(model=MODEL, file=file)
    return (result.text or "").strip()


if __name__ == "__main__":
    with open(sys.argv[1], "rb") as f:
        print(transcribe(f.read(), sys.argv[1]))
