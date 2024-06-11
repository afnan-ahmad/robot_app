import sounddevice as sd

# Define parameters
DURATION = 5  # seconds
SAMPLE_RATE = 44100  # Hz

while True:
    # Record audio
    print("Recording...")
    audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")

    # Play back the recording
    print("Playing back the recording...")
    sd.play(audio_data, samplerate=SAMPLE_RATE)
    sd.wait()  # Wait until playback is finished
    print("Playback finished.")
