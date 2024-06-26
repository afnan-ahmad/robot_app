import stretch_body.robot

r = stretch_body.robot.Robot()

did_startup = r.startup()
print(f'Robot connected to hardware: {did_startup}')

is_homed = r.is_homed()
print(f'Robot is homed: {is_homed}')


# Tilt the camera up and down
r.head.move_to('head_tilt', 0.49) # Up maximum tilt
r.push_command()
r.wait_command()

r.head.move_to('head_tilt', -0.49) # Down maximum tilt
r.push_command()
r.wait_command()

# Pan the camera left and right

r.head.move_to('head_pan', -.95) # left maximum pan
r.push_command()
r.wait_command()

r.head.move_to('head_pan', -2.18) # right maximum pan
r.push_command()
r.wait_command()

r.head.move_to('head_pan',-1.74) # center
r.push_command()
r.wait_command()