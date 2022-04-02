import os
import sys
import csv
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

def go_to_init_position(robot):
    robot.update_joint_states()
    # init_pose = robot.fk_pose(robot.position)
    # init_pose[0] = 0.5
    # init_pose[1] = 0
    # init_pose[2] = 0.35
    # init_pose[3] = np.sqrt(2)/2
    # init_pose[4] = -init_pose[3]
    # init_pose[5] = 0
    # init_pose[6] = 0
    pi = np.pi
    # new_angles = robot.ik_pose(init_pose[:3], init_pose[3:])
    new_angles = np.zeros(6)
    new_angles[1] = -pi/2
    new_angles[2] = pi/2
    new_angles[3] = -pi/2
    new_angles[4] = -pi/2
    robot.set_state(state = np.array(new_angles), value = .04)
    print('Get ready.')
    print(new_angles)
    while (np.abs(new_angles - robot.position)).sum() > 1e-2:
        robot.step()
    init_pose = robot.fk_pose(robot.position)
    print('Start...')
    return init_pose[3:]


def Xo(t: float, start_pose):
    X = start_pose[0] + 0.1*np.sin(0.05*t)
    Y = start_pose[1] + 0.1*np.cos(0.05*t)
    return np.array([X, Y, .0951])




def main():

    table, box = add_objects_to_scene()
    robot = UR5eSim(realtime = False)
    robot.initialize()
    time.sleep(2)
    des_force = 15
    sp = SimplePlot(time_limit = 5, legend = ['Fx','Fy', 'Fz', 'Tx', 'Ty', 'Tz'])
    ROT = go_to_init_position(robot)
    start_state = [.5, 0]
    desire_force = np.array([.0, .0, 35, .0, .0, .0])
    start_des_forces = desire_force.copy()
    Kp = np.array([0,0,1])*1e-8
    Ki = np.array([0,0,1])*1e-4

    t = 0
    sum_eF = np.zeros(3)
    last_time = time.time()
    output = []
    with open('trj_6.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile)
        while t < 200:
            line = []
            line.append(t)
            ## Game with time
            new_last_time = last_time
            last_time = time.time()
            dt = last_time - new_last_time
            t += dt
            desire_force[2] = start_des_forces[2] + 10*np.sin(t*2)
            line += desire_force.tolist()
            # print(desire_force)
            # robot.update_joint_states()
            pose = robot.fk_pose(robot.position)
            xo = Xo(t, start_state)
            line += xo.tolist()
            force_vector = robot.FT_sensor[:3] + np.array([0,0,3])
            line += force_vector.tolist()
            if (force_vector).sum() < .5:
                sum_eF += np.array([0,0, 0.001])
                # print('Hello')
                eF = np.zeros(3)
            else:
                eF = desire_force[:3] - force_vector
                sum_eF += eF*2e-2
            xf = Kp*eF + Ki*sum_eF
            xd = xo - xf
            line += xd.tolist()
            line += pose[:3].tolist()
            print(robot.jacobian.dot(robot.velocity))
            # print(xd)
            new_state = robot.ik_pose(position = xd, orientation = ROT)

            robot.set_state(state = np.array(new_state), value = .5)
            robot.step()
            sp.update((robot.FT_sensor + np.array([0,0,3,0,0,0])).tolist()+[desire_force[2]])
            spamwriter.writerow(line)
            # print(np.linalg.inv(robot.jacobian(robot.position).T))


if __name__=='__main__':
    main()
