from dotenv import load_dotenv

load_dotenv()

import logging
import sys

from nlu import RobotNLU
from transcribe import RobotASR

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

robot_nlu = RobotNLU()
robot_asr = RobotASR()

while True:
    print('Recognizing from microphone... ')
    text = robot_asr.recognize_from_mic(5)
    print('Recognized text:', text)

    obj = robot_nlu.extract_object_and_color(text)

    yolo_class = robot_nlu.find_most_similar_class(obj)
    
    print(obj, yolo_class)