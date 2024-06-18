import pyrealsense2 as rs
import numpy as np
import cv2
import stretch_body.robot

r = stretch_body.robot.Robot()
did_startup = r.startup()
print(f'Robot connected to hardware: {did_startup}')

is_homed = r.is_homed()
print(f'Robot is homed: {is_homed}')

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

def move_camera_to_center(x, y, frame_width, frame_height, threshold=50):
    center_x = frame_width // 2
    center_y = frame_height // 2

    min_x = center_x - threshold
    max_x = center_x + threshold
    min_y = center_y - threshold
    max_y = center_y + threshold

    if min_x < x < max_x and min_y < y < max_y:
        return  

    pan_movement = (center_x - x) / frame_width * 1.73 * 2  
    tilt_movement = (center_y - y) / frame_height * 0.49 * 2  

    r.head.get_joint.move_by('head_pan', pan_movement)
    r.head.get_joint.move_by('head_tilt', tilt_movement)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        x = 320  
        y = 240  

        cv2.circle(color_image, (x, y), 5, (0, 255, 0), -1)

    
        frame_height, frame_width, _ = color_image.shape
        move_camera_to_center(x, y, frame_width, frame_height)

        cv2.imshow('Tracking', color_image)
        if cv2.waitKey(1) == 27:
            break

finally:
    pipeline.stop()
    r.stop()
    cv2.destroyAllWindows()
