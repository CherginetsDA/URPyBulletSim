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
        base_position = np.array([0.7, 0, 0])
    )

    return table, box


def main():

    table, box = add_objects_to_scene()
    robot = UR5eSim()
    robot.initialize()
    robot.add_gui_sliders()
    sp = SimplePlot(time_limit = 5, legend = ['Fx','Fy', 'Fz', 'Tx', 'Ty', 'Tz'])
    # sp_2 = SimplePlot(time_limit = 5, legend = ['Fx','Fy', 'Fz', 'Tx', 'Ty', 'Tz'])
    while True:
        robot.set_state(state = robot.read_gui_sliders())
        robot.update_joint_states()
        pose = robot.fk_pose(robot.position)
        CP = pose[:3]
        ROT = pose[3:]
        print(CP)
        print(ROT)
        new_angles = robot.ik_pose(CP, ROT)
        # new_angles = robot.calculate_ik(np.array(CP),ROT, isEuler = False)
        print('Angles')
        print(robot.position.tolist())
        print(new_angles.tolist())
        # sp.update(robot.position)
        # sp.update(robot.velocity)
        # sp_2.update(np.dot(robot.jacobian(robot.position).T,robot.FT_sensor))
        # sp.update(robot.torque)
        # print(robot.FT_sensor)
        sp.update(robot.FT_sensor)


if __name__=='__main__':
    main()
