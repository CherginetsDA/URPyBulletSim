import PyBulletClient as pbc
import os
import time
import numpy as np
from collections import namedtuple
from attrdict import AttrDict

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
            0.0,
            -np.pi/2,
            0.0
        ]),
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
        time.sleep(0.25)
        self.update_joint_states()



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

    def jacobian(self, position: np.array = None):
        '''
        It should get the jacobian, bun it doesn't work

        Keyword arguments:
        state --    position for calculate jacobian, if it is None, takes current
                    robot position
        '''
        if position is None:
            self.update_joint_states()
            state = self.position
        else:
            state = position
        return pbc.get_jacobian(
            body_id = self.__body_id,
            num_joints = self.num_joints,
            position = state
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
        self.__FT_sensor = np.array(temp[-1][2])

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
            'ur5e.urdf'
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
