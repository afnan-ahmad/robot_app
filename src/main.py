from dotenv import load_dotenv

load_dotenv()

from nlu import RobotNLU
from transcribe import RobotASR

robot_nlu = RobotNLU()
robot_asr = RobotASR()

while True:
    print('Recognizing from microphone... ')
    text = robot_asr.recognize_from_mic(5)
    print('Recognized text:', text)

    obj, color = robot_nlu.extract_object_and_color(text)
    
    print(obj, color)