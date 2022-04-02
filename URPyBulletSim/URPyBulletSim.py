from . import PyBulletClient as pbc
from . import jacobi_der
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
        base_orientation: np.ndarray = np.array([0., 0., 0., 1.]),
        realtime: bool = True
    ):
        if realtime:
            pbc.set_real_time()
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
        self.__FT_sensor = np.zeros(6)
        self.__filter_N = 1
        self.__ft_buffer = np.zeros((6,self.__filter_N))
        self.__ft_index = 0

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
        self.__dyn_params       = kdl.ChainDynParam(self.__chain,kdl.Vector(0,0,-9.81))

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

    @property
    def jacobian(self) -> np.ndarray:
        jac = kdl.Jacobian(self.__position.shape[0])
        self.__jacsolver.JntToJac(to_jnt_array(self.__position), jac)
        return to_np_matrix(jac, self.__position.shape[0])

    @property
    def D_jacobian(self) -> np.ndarray:
        d_jac = kdl.Jacobian(self.__position.shape[0])
        self.__djacsolver.JntToJacDot(to_jnt_array_vel(self.__position, self.__velocity), d_jac)
        return to_np_matrix(d_jac, self.__position.shape[0])
    ## TODO: now jacobian derivative just for ur5e.
    # Maybe getting parameters from pybullet will be a good idea
    @property
    def d_jacobian(self) -> np.ndarray:
        return jacobi_der.der_jacobian(self.__position, self.__velocity)

    @property
    def mass_matrix(self) -> np.ndarray:
        return pbc.get_mass_matrix(
            body_id = self.__body_id,
            position = self.__position
        )

    @property
    def Mass_matrix(self) -> np.ndarray:
        mm = kdl.JntSpaceInertiaMatrix(self.__position.shape[0])
        self.__dyn_params.JntToMass(to_jnt_array(self.__position), mm)
        return to_np_matrix(mm, self.__position.shape[0])

    @property
    def Coriolis_vector(self) -> np.ndarray:
        cv = kdl.JntArray(len(self.__index_control_joints))
        self.__dyn_params.JntToCoriolis(to_jnt_array(self.__position),to_jnt_array(self.__velocity), cv)
        return to_np_matrix(cv, self.__position.shape[0])

    @property
    def Gravity_vector(self) -> np.ndarray:
        gv = kdl.JntArray(len(self.__index_control_joints))
        self.__dyn_params.JntToGravity(to_jnt_array(self.__position), gv)
        return to_np_matrix(gv, self.__position.shape[0])

    @property
    def coriolis_and_gravity_vector(self) -> np.ndarray:
        return pbc.get_coriolis_and_gravity_vector(
            body_id = self.__body_id,
            position = self.__position,
            velocity = self.__velocity
        )

    def fk_pose(self, q: np.ndarray, quaternion: bool = True):
        end_frame = kdl.Frame()
        self.__fk_posesolver.JntToCart(to_jnt_array(q),end_frame)
        pos = end_frame.p
        rot = kdl.Rotation(end_frame.M)
        if quaternion:
            rot = rot.GetQuaternion()
            return np.array([pos[0], pos[1], pos[2],
                             rot[0], rot[1], rot[2], rot[3]])
        else:
            rot = rot.GetEulerZYX()
            return np.array([pos[0], pos[1], pos[2],
                             rot[0], rot[1], rot[2]])

    def rot(self, q: np.ndarray) -> np.ndarray:
        end_frame = kdl.Frame()
        T = self.__fk_posesolver.JntToCart(to_jnt_array(q), end_frame)
        rot = to_np_matrix(kdl.Rotation(end_frame.M), 3)
        return rot


    def twist(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        vel_frame = kdl.FrameVel()
        self.__fk_velsolver.JntToCart(to_jnt_array_vel(q, dq), vel_frame)
        return to_np_matrix(vel_frame.GetTwist(),6)

    def set_state(self, state: np.ndarray, value):
        '''Kp =
        Set desire position and robot get it by the Position control.

        Keyword arguments:
        state -- desire position
        '''
        pbc.set_states(
            body_id = self.__body_id,
            control_joints = self.__index_control_joints,
            state = state,
            value = value
        )

    def step(self):
        pbc.step()
        self.update_joint_states()

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
        ft_sensor = np.array(
            pbc.get_joint_states(self.__body_id, [self.num_joints - 2])[0][2]
        )
        ## Корректировка на величину веса
        ft_sensor[:3] -= self.rot(self.__position).T.dot(
                np.array([0,0,3-0.056]).reshape(3,1)
            ).flatten()
        for i in range(6):
            sign = np.sign(ft_sensor[i])
            if ft_sensor[i]*sign > 100:
                ft_sensor[i] = 100*sign
            self.__ft_buffer[i][self.__ft_index] = ft_sensor[i]
            self.__FT_sensor[i] = self.__ft_buffer[i].mean()
        self.__ft_index = (self.__ft_index + 1) % self.__filter_N
        # pose = self.fk_pose(self.__position)
        # temp = kdl.Rotation().Quaternion(pose[3],pose[4],pose[5],pose[6])
        # ## TODO: fix mass
        # v = kdl.Vector(0,0,3)
        # res = temp*v
        # self.__FT_sensor[0] += res[1]
        # self.__FT_sensor[1] += res[0]
        # self.__FT_sensor[2] -= res[2]



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

    def impedance_control_signals(
        self,
        pos_d: np.ndarray,
        vel_d: np.ndarray,
        acc_d: np.ndarray,
        fd: bool = False,
        cp: bool = False
    ) -> np.ndarray:
        Ms = np.diag([1]*6)*10 # inertia of spring
        Bs = np.diag([800, 800, 800, 80, 80, 80])*0.01 # damping of spring
        Ks = np.diag([80, 80, 200, 20, 20, 20]) # rigidity of spring
        J = self.jacobian.reshape(6,6)
        JT = J.T
        invJ = np.linalg.inv(J)
        invJT= invJ.T
        # M = self.mass_matrix.reshape(6,6)
        M = self.Mass_matrix.reshape(6,6)
        # H = (self.Coriolis_vector + self.Gravity_vector).reshape(6,1)
        H = self.coriolis_and_gravity_vector.reshape(6,1)
        Gamma = invJT.dot(M).dot(invJ)
        # np.dot(invJT, M, invJ)

        h = invJT.dot(H) - Gamma.dot(self.D_jacobian).dot(self.velocity.reshape(6,1))
        Fa = self.FT_sensor.reshape(6,1)
        ## Check it
        temp = (invJT.dot((H - JT.dot(Fa))) + Fa - h).flatten()
        # print('Must be around zero: ',temp)
        # print((self.__torque - H.flatten()))
        # ddq = np.linalg.inv(M).dot(self.__torque.reshape(6,1) + JT.dot(Fa) - H)
        # print(ddq.flatten())

        pose = self.fk_pose(
            q = self.__position.reshape(6,1),
            quaternion = False
        )
        pose_err = -pose.reshape(6,1) + pos_d.reshape(6,1)
        if pose_err.flatten()[5] > np.pi:
            pose_err[5] -=2*np.pi
        if pose_err.flatten()[5] < -np.pi:
            pose_err[5] +=2*np.pi
        y = acc_d.reshape(6,1) + \
            np.linalg.inv(Gamma).dot(
                Bs.dot(vel_d.reshape(6,1) - J.dot(self.__velocity.reshape(6,1))) +\
                Ks.dot(pose_err) +\
                Fa
            )
        # print('Impedance: ', y.flatten())
        ctrl = (J.T).dot(Gamma.dot(y) + h - Fa)
        if fd :
            if cp:
                ddx = np.linalg.inv(Gamma).dot(invJT.dot(ctrl) - h)
                return y

            ddq = np.linalg.inv(M).dot(ctrl + JT.dot(Fa*0) - H)
            return ddq
        print("STSUKO")
        # ddq = np.linalg.inv(M).dot(ctrl + JT.dot(Fa) - H)
        # ddq = np.linalg.inv(M).dot(ctrl + JT.dot(Fa)*0 - H)
        # print('AAAA', (J.dot(ddq) - y + self.D_jacobian.dot(self.velocity.reshape(6,1))).flatten())
        # print(np.linalg.inv(Gamma).dot(invJ.T.dot(u) - h + Fa))
        return ctrl

    def fd(
        self,
        ctrl : np.ndarray
    ) -> np.ndarray:
        J = self.jacobian
        invJ = np.linalg.inv(J)
        M = self.Mass_matrix
        H = (self.Coriolis_vector + self.Gravity_vector).reshape(6,1)
        Gamma = np.dot(invJ.T, M, invJ)
        h = (invJ.T).dot(H) - (Gamma.dot(self.D_jacobian)).dot(self.velocity.reshape(6,1))
        # ddq = np.linalg.inv(M).dot(ctrl + JT.dot(Fa) - H)
        ddx = np.linalg.inv(Gamma).dot(invJ.T.dot(ctrl) - h)
        # print(np.linalg.det(M))
        # print(np.linalg.inv(Gamma))
        # print('Forward D: ', ddx.flatten())
        return ddx



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
        realtime: bool = True
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
            base_orientation = base_orientation,
            realtime = realtime
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
