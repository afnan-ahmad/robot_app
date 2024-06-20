import stretch_body.robot

r = stretch_body.robot.Robot()

did_startup = r.startup()
print(f'Robot connected to hardware: {did_startup}')

is_homed = r.is_homed()
print(f'Robot is homed: {is_homed}')

location = 3

if location == 1:
    r.arm.move_to(0.3) 
    r.lift.move_to(0.8)
    r.push_command()
    r.wait_command()

    r.end_of_arm.move_to('wrist_pitch', 0)
    r.end_of_arm.move_to('wrist_yaw', -.80)
    r.push_command()
    r.wait_command()

    r.arm.move_to(0.4) 
    r.push_command()
    r.wait_command()

    r.end_of_arm.move_to('stretch_gripper', 60)
    r.push_command()
    r.wait_command()

    r.arm.move_to(0.43) 
    r.push_command()
    r.wait_command()

    r.end_of_arm.move_to('stretch_gripper', 25)
    r.push_command()
    r.wait_command()

    r.lift.move_to(0.9)
    r.push_command()
    r.wait_command()

if location == 3:
    r.arm.move_to(0.48) 
    r.lift.move_to(0.8)
    r.push_command()
    r.wait_command()

    r.end_of_arm.move_to('wrist_pitch', 0)
    r.end_of_arm.move_to('wrist_yaw', -0.4)
    r.push_command()
    r.wait_command()

    r.end_of_arm.move_to('stretch_gripper', 80)
    r.push_command()
    r.wait_command()

    r.end_of_arm.move_to('stretch_gripper', 20)
    r.push_command()
    r.wait_command()

    r.lift.move_to(1.3)
    r.push_command()
    r.wait_command()

r.stop()