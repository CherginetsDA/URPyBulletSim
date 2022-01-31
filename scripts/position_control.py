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
    init_state = robot.position
    new_state = init_state.copy()
    # sp = SimplePlot(time_limit = 5, legend = ['Fx','Fy', 'Fz', 'Tx', 'Ty', 'Tz'])
    sp_2 = SimplePlot(time_limit = 5, legend = ['Fx','Fy', 'Fz', 'Tx', 'Ty', 'Tz'])
    while True:
        new_state[0] = init_state[0] + np.cos(time.time())
        robot.set_state(new_state)
        robot.update_joint_states()
        # sp.update(robot.position)
        # sp.update(robot.velocity)
        sp_2.update(np.dot(robot.jacobian(robot.position),robot.velocity))
        # sp.update(robot.torque)
        # sp.update(robot.FT_sensor)


if __name__=='__main__':
    main()
