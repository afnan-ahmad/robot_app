import stretch_body.robot

r = stretch_body.robot.Robot()

did_startup = r.startup()
print(f'Robot connected to hardware: {did_startup}')

is_homed = r.is_homed()
print(f'Robot is homed: {is_homed}')


# Tilt the camera up and down
r.head.get_joint.move_to('head_tilt', 0.49) # Up maximum tilt
r.head.get_joint.move_to('head_tilt', -0.49) # Down maximum tilt

# Pan the camera left and right
r.head.get_joint.move_to('head_pan', 1.73) # Left maximum pan
r.head.get_joint.move_to('head_pan', -1.73) # Right maximum pan