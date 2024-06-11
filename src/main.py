from nlu import RobotNLU
from transcribe import RobotASR

robot_nlu = RobotNLU()

robot_asr = RobotASR()

print(robot_asr.recognize_from_mic(5))

obj, color = robot_nlu.extract_object_and_color("Pick up the red ball from the table")

print(obj, color)