class ASRInterface:
    def recognize_from_mic(self, duration, channels) -> None:
        pass

    def audio_to_text(self, audio_data) -> str:
        pass