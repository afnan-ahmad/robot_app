from transcribe.interface import ASRInterface
import sounddevice as sd
import numpy as np
import vosk
import json
import os

class VoskASR(ASRInterface):
    sample_rate = None
    model_path = None
    recognizer = None

    def __init__(self, sample_rate=44100, model_path=None) -> None:
        self.sample_rate = sample_rate

        if model_path:
            self.model_path = model_path
        else:
            if 'VOSK_MODEL_PATH' in os.environ:
                self.model_path = os.getenv('VOSK_MODEL_PATH')
            else:
                raise Exception("VOSK_MODEL_PATH not specified in environment.")

        vosk.SetLogLevel(-1)

        self.recognizer = vosk.KaldiRecognizer(vosk.Model(self.model_path), self.sample_rate)

    def audio_to_text(self, audio_data) -> str:
        self.recognizer.AcceptWaveForm(audio_data)

        result = json.loads(self.recognizer.FinalResult())

        text = result['text']

        return text

    def recognize_from_mic(self, duration=5, channels=2) -> str:
        audio_data = sd.rec(int(duration * self.sample_rate), samplerate=self.sample_rate, channels=channels, dtype='int16')

        sd.wait()

        return self.audio_to_text(np.array(audio_data).tobytes())
