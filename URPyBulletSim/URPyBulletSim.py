from . import PyBulletClient as pbc
import os
import sys
import time
import numpy as np
from collections import namedtuple
from attrdict import AttrDict

import spatialmath.base as spmb
import PyKDL as kdl

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from kdl_parser.kdl_parser_py.kdl_parser_py import urdf


class URSim:
    def __init__(
        self,
        urdf_path: str,
        base_position: np.ndarray = np.array([0., 0., 0.]),
        base_orientation: np.ndarray = np.array([0., 0., 0., 1.])
    ):
        self.__urdf_filepath = urdf_path
        self.__base_position = base_position
        self.__base_orientation = base_orientation
        self.__body_id = None
        self.num_joints = None
        self.__joints = None
        self.__index_control_joints = None
        self.__sliders = None
        self.__position = None
        self.__velocity = None
        self.__torque = None
        self.__FT_sensor = None

    def initialize(
        self,
        init_position: np.ndarray = np.array([
            0.0,
            -np.pi/2,
            np.pi/2,
            -np.pi/2,
            -np.pi/2,
            0.0
        ])
    ):
        '''
        initialize the robot.

        Keyword arguments:
        init_position -- you can setup init position of robot
        '''
        self.__body_id = pbc.add_object(
            urdf_filepath = self.__urdf_filepath,
            base_position = self.__base_position,
            base_orientation = self.__base_orientation
        )

        self.num_joints = pbc.get_joint_number(self.__body_id)
        self.__index_control_joints = []
        for i in range(self.num_joints):
            info = pbc.get_joint_info(body_id = self.__body_id, joint_id = i)
            if info[2] == pbc.p.JOINT_REVOLUTE:
                self.__index_control_joints.append(info[0])
                pbc.init_joint_control(
                    body_id = self.__body_id,
                    joint_id = info[0],
                    target = init_position[len(self.__index_control_joints) - 1]
                )
            pbc.init_force_torque_sensor(
                body_id = self.__body_id,
                joint_id = info[0]
            )
        self.__initialize_kdl()
        time.sleep(0.5)
        self.update_joint_states()


    def __initialize_kdl(self):
        urdf_file = open(self.__urdf_filepath, 'r')
        urdf_str = urdf_file.read()
        urdf_file.close()
        # Generate kinematic model for orocos_kdl
        (ok, self.__tree) = urdf.treeFromString(urdf_str)

        if not ok:
            raise RuntimeError('Tree is not valid')

        self.__chain = self.__tree.getChain(
            pbc.p.getBodyInfo(self.__body_id)[0].decode('UTF-8'),
            pbc.get_joint_info(
                self.__body_id, self.num_joints - 2
            )[12].decode('UTF-8')
        )

        self.__fk_posesolver    = kdl.ChainFkSolverPos_recursive(self.__chain)
        self.__fk_velsolver     = kdl.ChainFkSolverVel_recursive(self.__chain)
        self.__ik_velsolver     = kdl.ChainIkSolverVel_pinv(self.__chain)
        self.__ik_posesolver    = kdl.ChainIkSolverPos_NR(self.__chain, self.__fk_posesolver, self.__ik_velsolver)

        self.__jacsolver        = kdl.ChainJntToJacSolver(self.__chain)
        self.__djacsolver       = kdl.ChainJntToJacDotSolver(self.__chain)

    def ik_pose(self, position: np.ndarray,  orientation = None):
        pos = kdl.Vector(position[0],position[1],position[2])
        if not orientation is None:
            rot = kdl.Rotation()
            rot = rot.Quaternion(
                orientation[0],
                orientation[1],
                orientation[2],
                orientation[3]
            )
            goal = kdl.Frame(rot, pos)
        else:
            goal = kdl.Frame(pos)
        result = kdl.JntArray(len(self.__index_control_joints))
        joints_angles = to_jnt_array(self.__position)

        self.__ik_posesolver.CartToJnt(joints_angles, goal, result)
        return np.array(list(result))


    def jacobian(self, q: np.ndarray) -> np.ndarray:
        jac = kdl.Jacobian(q.shape[0])
        self.__jacsolver.JntToJac(to_jnt_array(q), jac)
        return to_np_matrix(jac, q.shape[0])

    def fk_pose(self, q: np.ndarray):
        end_frame = kdl.Frame()
        self.__fk_posesolver.JntToCart(to_jnt_array(q),end_frame)
        pos = end_frame.p
        rot = kdl.Rotation(end_frame.M)
        rot = rot.GetQuaternion()
        return np.array([pos[0], pos[1], pos[2],
                         rot[0], rot[1], rot[2], rot[3]])

    def rot(self, q: np.ndarray) -> np.ndarray:
        end_frame = kdl.Frame()
        T = self.__fk_posesolver.JntToCart(to_jnt_array(q), end_frame)
        rot = to_np_matrix(kdl.Rotation(end_frame.M), 3)
        return rot


    def twist(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        vel_frame = kdl.FrameVel()
        self.__fk_velsolver.JntToCart(to_jnt_array_vel(q, dq), vel_frame)
        return to_np_matrix(vel_frame.GetTwist(),6)

    def set_state(self, state: np.ndarray):
        '''
        Set desire position and robot get it by the Position control.

        Keyword arguments:
        state -- desire position
        '''
        pbc.set_states(
            body_id = self.__body_id,
            control_joints = self.__index_control_joints,
            state = state
        )

    def add_gui_sliders(self):
        '''
        This add sliders to simulation to easy setting position for robot
        '''
        self.__sliders = []
        for i in range(len(self.__index_control_joints)):
            self.__sliders.append(
                pbc.get_user_debug_parameter(
                    'q_'+str(i),
                    (-np.pi, np.pi),
                    self.position[i]
                )
            )

    def read_gui_sliders(self):
        '''
        Reading slider states
        '''
        return np.array([pbc.read_user_debag_parameter(self.__sliders[i]) \
            for i in range(len(self.__index_control_joints))])

    def update_joint_states(self):
        '''
        Update robot state from simulation. You can get position, velocity,
        torque and FT_sensor as a property of the object
        '''
        temp = pbc.get_joint_states(self.__body_id, self.__index_control_joints)
        self.__position = np.array([i[0] for i in temp])
        self.__velocity = np.array([i[1] for i in temp])
        self.__torque = np.array([i[3] for i in temp])
        self.__FT_sensor = np.array(
            pbc.get_joint_states(self.__body_id, [self.num_joints - 2])[0][2]
        )


    # def calculate_ik(
    #     self,
    #     position,
    #     orientation,
    #     current_angles = None
    # ):
    #     quaternion = pbc.get_quaternion_from_euler(orientation)
    #     lower_limits = []
    #     upper_limits = []
    #     joint_ranges = []
    #     joint_damping = []
    #     if current_angles is None:
    #         rest_poses = [0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]
    #     else:
    #         rest_poses = current_angles
    #
    #     for index in self.__index_control_joints:
    #         temp = pbc.get_joint_info(self.__body_id, index)
    #         joint_damping.append(temp[6])
    #         lower_limits.append(temp[8])
    #         upper_limits.append(temp[9])
    #         joint_ranges.append(upper_limits[-1] - lower_limits[-1])
    #
    #     return pbc.p.calculateInverseKinematics(
    #         self.__body_id,
    #         self.num_joints - 2,
    #         position.tolist(),
    #         quaternion.tolist(),
    #         jointDamping=joint_damping,
    #         upperLimits=upper_limits,
    #         lowerLimits=lower_limits,
    #         jointRanges=joint_ranges,
    #         restPoses=rest_poses
    #     )

    @property
    def position(self):
        return self.__position

    @property
    def velocity(self):
        return self.__velocity

    @property
    def torque(self):
        return self.__torque

    @property
    def FT_sensor(self):
        return self.__FT_sensor


def to_np_matrix(kdl_data, size: int) -> np.ndarray:
    if isinstance(kdl_data, (kdl.JntSpaceInertiaMatrix, kdl.Jacobian, kdl.Rotation)):
        out = np.zeros((size, size))
        for i in range(0, size):
            for j in range(0, size):
                out[i,j] = kdl_data[i,j]
        return out
    elif isinstance(kdl_data, (kdl.JntArray, kdl.JntArrayVel)):
        out = np.zeros(size)
        for i in range(0, size):
            out[i] = kdl_data[i]
        return out
    else:
        out = np.zeros(size)
        for i in range(0, size):
            out[i] = kdl_data[i]
        return out

def to_jnt_array(np_vector: np.ndarray)-> kdl.JntArray:
    size = np_vector.shape[0]
    ja = kdl.JntArray(size)
    for i in range(0, size):
        ja[i] = np_vector[i]
    return ja

def to_jnt_array_vel(q: np.ndarray, dq:np.ndarray)-> kdl.JntArrayVel:
    size = q.shape[0]
    jav = kdl.JntArrayVel(size)
    jav.q = to_jnt_array(q)
    jav.qdot = to_jnt_array(dq)
    return jav

def quaternioun(matrix: np.ndarray) -> np.ndarray:
    pass

class UR5eSim(URSim):
    '''
    Class UR5e simulation
    '''
    def __init__(
        self,
        base_position: np.ndarray = np.array([0., 0., 0.]),
        base_orientation: np.ndarray = np.array([0., 0., 0., 1.]),
        ):
        urdf_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data',
            'ur_e_description',
            'urdf',
            'ur5e_with_block.urdf'
        )

        super().__init__(
            urdf_path = urdf_path,
            base_position = base_position,
            base_orientation = base_orientation
        )


if __name__=='__main__':
    import time
    import pybullet_data
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from SimplePlot.SimplePlot import SimplePlot

    table_urdf_path = os.path.join(
        pybullet_data.getDataPath(),
        'table',
        'table.urdf'
    )

    table = pbc.add_object(
        urdf_filepath = table_urdf_path,
        base_position = np.array([0.5, 0, -0.6300])
    )
    del table_urdf_path

    sp = SimplePlot(time_limit = 5)

    robot = UR5eSim()
    robot.initialize()
    robot.add_gui_sliders()

    while True:
        robot.set_state(state = robot.read_gui_sliders())
        robot.update_joint_states()
        # sp.update(robot.position)
        # sp.update(robot.velocity)
        # sp.update(robot.torque)
        # sp.update(robot.FT_sensor)
        print(robot.jacobian())
