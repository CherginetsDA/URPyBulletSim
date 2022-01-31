import os
import sys
import time
import pybullet_data
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SimplePlot import SimplePlot
from URPyBulletSim import UR5eSim
from URPyBulletSim import PyBulletClient as pbc


def add_objects_to_scene():
    table_urdf_path = os.path.join(
        pybullet_data.getDataPath(),
        'table',
        'table.urdf'
    )

    table = pbc.add_object(
        urdf_filepath = table_urdf_path,
        base_position = np.array([0.5, 0, -0.6300])
    )

    box_urdf_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data',
        'obstacles',
        'urdf',
        'small_box.urdf'
    )

    box = pbc.add_object(
        urdf_filepath = box_urdf_path,
        base_position = np.array([0.5, 0, 0])
    )

    return table, box

def go_to_init_position(robot):
    robot.update_joint_states()
    init_pose = robot.fk_pose(robot.position)
    init_pose[0] = 0.5
    init_pose[1] = 0
    init_pose[2] = 0.3
    new_angles = robot.ik_pose(init_pose[:3], init_pose[3:])
    robot.set_state(state = np.array(new_angles))
    print('Get ready.')
    time.sleep(1)
    print('Start...')
    return init_pose[3:]





def main():

    table, box = add_objects_to_scene()
    robot = UR5eSim()
    robot.initialize()
    time.sleep(2)
    des_force = 15
    sp = SimplePlot(time_limit = 5, legend = ['Fx','Fy', 'Fz', 'Tx', 'Ty', 'Tz'])
    ROT = go_to_init_position(robot)
    while True:
        robot.update_joint_states()
        pose = robot.fk_pose(robot.position)
        CP = pose[:3]
        err_sum = 0
        err = (des_force - robot.FT_sensor[2])
        dz = err*0.001
        if np.abs(dz) > 5e-5:
            dz = 5e-5*np.sign(dz)
        CP[2] -= dz
        new_angles = robot.ik_pose(CP, ROT)
        robot.set_state(state = np.array(new_angles))
        sp.update(robot.FT_sensor)

if __name__=='__main__':
    main()
