from transcribe.vosk import VoskASR

from logging import getLogger

logger = getLogger("RobotASR")

class RobotASR:
    engine = None

    def __init__(self, provider='vosk') -> None:

        logger.info("Initializing Robot ASR...")

        self.engine = VoskASR()

    def audio_to_text(self, audio_data) -> str:
        return self.engine.audio_to_text(audio_data)
    
    def recognize_from_mic(self, duration=5, channels=2) -> str:
        return self.engine.recognize_from_mic(duration=duration, channels=channels)