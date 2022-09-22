import whisper
from io import BytesIO

from pydub import AuAudioSegment
AudioSegment.converter = which("ffmpeg")

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model

    model = whisper.load_model("tiny")

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    bytes = model_inputs.get('audio_bytes', None)
    if bytes:
        audio_obj = AuAudioSegment.from_file(BytesIO(bytes))
        audio_obj.export("local.mp3", format="mp3")

        result = model.transcribe("local.mp3")
        return result

    return {"error": "something went wrong"}
