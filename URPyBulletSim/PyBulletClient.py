import numpy as np
import pybullet as p
from typing import List
from pybullet_utils import bullet_client as bc
from pybullet_utils import urdfEditor as ed

client = bc.BulletClient(connection_mode = p.GUI)
p.setGravity(gravX = 0, gravY = 0, gravZ = 9.81)
p.setRealTimeSimulation(True)


def add_object(
    urdf_filepath: str,
    base_position: np.ndarray = np.array([0., 0., 0.]),
    base_orientation: np.ndarray = np.array([0., 0., 0., 1.])
) -> int:
    '''
    Add the object to pybullet scene

    Keyword arguments:
    urdf_filename -- the file name of the object URDF file
    base_position -- base position of the object

    return: body_num
    '''
    return get_client().loadURDF(
        fileName = urdf_filepath,
        basePosition = base_position,
        baseOrientation = base_orientation,
        flags = p.URDF_USE_SELF_COLLISION\
         | p.URDF_INITIALIZE_SAT_FEATURES\
         | p.URDF_USE_IMPLICIT_CYLINDER
         )

def get_client():
    return globals()['client']

def get_joint_number(body_id: int) -> int:
    return get_client().getNumJoints(body_id)

def get_joint_info(body_id: int, joint_id: int) -> list:
    return get_client().getJointInfo(body_id, joint_id)

def init_joint_control(body_id: int, joint_id: int, target: float = 0.0):
    p.setJointMotorControl2(
        bodyUniqueId = body_id,
        jointIndex = joint_id,
        controlMode = p.POSITION_CONTROL,
        targetPosition = target,
        targetVelocity = 0,
        force = 10000
    )
    p.enableJointForceTorqueSensor(
        bodyUniqueId = body_id,
        jointIndex = joint_id,
        enableSensor = True
    )

def set_states(body_id: int, control_joints: List[int], state: np.ndarray):
    forces = []
    for i in range(len(control_joints)):
        forces.append(
            get_joint_info(body_id = body_id, joint_id = control_joints[i])[10]
        )
    p.setJointMotorControlArray(
            body_id,
            control_joints,
            p.POSITION_CONTROL,
            targetPositions=state.tolist(),
            targetVelocities=[0]*len(state),
            positionGains=[0.04]*len(state),
            forces=forces
        )

def get_user_debug_parameter(name, limits, value):
    print(value)
    return p.addUserDebugParameter(name, min(limits), max(limits), value)

def read_user_debag_parameter(parameter):
    return p.readUserDebugParameter(parameter)

def get_joint_states(
    body_id: int,
    control_joints: List[int]
) -> List[List[float]]:
    return p.getJointStates(body_id, control_joints)

def get_jacobian(
    body_id: int,
    num_joints: int,
    position: np.ndarray,
    velocity: np.ndarray = None,
    acceleration: np.ndarray = None,
):
    n = len(position)
    if velocity is None:
        velocity = np.array([0]*n)
    if acceleration is None:
        acceleration = np.array([0]*n)
    jacobian = p.calculateJacobian(
                    bodyUniqueId = body_id,
                    linkIndex = num_joints - 1,
                    localPosition = [0]*3,
                    objPositions = position.tolist(),
                    objVelocities = velocity.tolist(),
                    objAccelerations = acceleration.tolist()
                )
    print(jacobian)
    return np.array(jacobian[0] + jacobian[1])
