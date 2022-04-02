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
import PyKDL as kdl


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
    init_pose = robot.fk_pose(robot.position)
    init_pose[0] = 0.5
    init_pose[1] = 0
    init_pose[2] = 0.1
    pi = np.pi
    new_angles = robot.ik_pose(init_pose[:3], init_pose[3:])
    # new_angles = np.zeros(6)
    # new_angles[1] = -pi/2
    # new_angles[2] = pi/2
    # new_angles[3] = -pi/2
    # new_angles[4] = -pi/2
    robot.set_state(state = np.array(new_angles), value = .04)
    print('Get ready.')
    print(new_angles)
    while (np.abs(new_angles - robot.position)).sum() > 1e-2:
        robot.step()
    init_pose = robot.fk_pose(robot.position,quaternion = False)
    print('Start...')
    return init_pose[3:]


def Xo(t: float, start_pose):
    X = start_pose[0] + 0.1*np.sin(0.05*t)
    Y = start_pose[1] + 0.1*np.cos(0.05*t)
    return np.array([0.5, 0.2, .4 ])


def main():
    np.set_printoptions(precision=3, suppress=True)
    table, box = add_objects_to_scene()
    robot = UR5eSim()
    robot.initialize()
    pose = robot.fk_pose(robot.position, quaternion = False)
    rot = go_to_init_position(robot)
    rot[2] += 0.1
    acc = np.array([0]*6)
    sp = SimplePlot(size_limit = 500, legend = ['Fx','Fy', 'Fz', 'Tx', 'Ty', 'Tz'])
    t = 0
    last_time = time.time()

    while t < 200:
        if np.sqrt(np.sum(robot.FT_sensor**2)) < 25:
            t += 1e-2
        pose = robot.fk_pose(robot.position, quaternion = False)
        # sp.update(pose.flatten().tolist()[:3])
        sp.update(robot.FT_sensor.tolist())
        # print([ 0.5, 0.2, 0.005 - 0.001*t] + rot.tolist())
        # print('Here',rot)
        ROT = (rot + np.array([0,0,0*0.5*np.sin(t)])).tolist()
        ddq = robot.impedance_control_signals(
            np.array([ 0.5, 0.2, 0.005 - 0.01*t] + ROT),
            np.array([0]*2 + [0.01] + [0]*2 + [0.5*np.cos(t)*0]),
            np.array([0]*2 + [0] + [0]*2 + [-0.5*np.sin(t)*0]),
            fd = True,
            cp = True
        )
        dv = 5e-3*(acc +  ddq.flatten())
        robot_vel = robot.jacobian.dot(robot.velocity.reshape(6,1)).flatten()
        temp_vel = robot_vel + dv
        acc = ddq.flatten()
        dp = 5e-3*(robot_vel + temp_vel)
        new_position = pose + dp
        orientation = kdl.Rotation().EulerZYX(new_position[3],new_position[4],new_position[5]).GetQuaternion()
        new_angles =robot.ik_pose(
            position=new_position[:3].flatten(),
            orientation = np.array([orientation[0],orientation[1],orientation[2],orientation[3]]))
        robot.set_state(state = new_angles, value = .5)
        robot.update_joint_states()


# def main():
#
#     table, box = add_objects_to_scene()
#     robot = UR5eSim(realtime = False)
#     robot.initialize()
#     time.sleep(2)
#     des_force = 15
#     sp = SimplePlot(time_limit = 5, legend = ['Fx','Fy', 'Fz', 'Tx', 'Ty', 'Tz'])
#     ROT = go_to_init_position(robot)
#     start_state = [.5, 0]
#     desire_force = np.array([.0, .0, 35, .0, .0, .0])
#     start_des_forces = desire_force.copy()
#     Kp = np.array([0,0,1])*1e-8
#     Ki = np.array([0,0,1])*1e-4
#
#     t = 0
#     sum_eF = np.zeros(3)
#     last_time = time.time()
#     output = []
#     with open('trj_6.csv', 'w') as csvfile:
#         spamwriter = csv.writer(csvfile)
#         while t < 200:
#             line = []
#             line.append(t)
#             ## Game with time
#             new_last_time = last_time
#             last_time = time.time()
#             dt = last_time - new_last_time
#             t += dt
#             desire_force[2] = start_des_forces[2] + 10*np.sin(t*2)
#             line += desire_force.tolist()
#             print(desire_force)
#             # robot.update_joint_states()
#             pose = robot.fk_pose(robot.position)
#             xo = Xo(t, start_state)
#             line += xo.tolist()
#             force_vector = robot.FT_sensor[:3] + np.array([0,0,3])
#             line += force_vector.tolist()
#             if (force_vector).sum() < .5:
#                 sum_eF += np.array([0,0, 0.001])
#                 print('Hello')
#                 eF = np.zeros(3)
#             else:
#                 eF = desire_force[:3] - force_vector
#                 sum_eF += eF*2e-2
#             xf = Kp*eF + Ki*sum_eF
#             xd = xo - xf
#             line += xd.tolist()
#             line += pose[:3].tolist()
#
#             print(xd)
#             new_state = robot.ik_pose(position = xd, orientation = ROT)
#
#             robot.set_state(state = np.array(new_state), value = .5)
#             robot.step()
#             sp.update((robot.FT_sensor + np.array([0,0,3,0,0,0])).tolist()+[desire_force[2]])
#             spamwriter.writerow(line)
#             # print(np.linalg.inv(robot.jacobian(robot.position).T))


if __name__=='__main__':
    main()
