from robot_properties_solo.solo12wrapper import Solo12Config, Solo12Robot
from pinocchio.robot_wrapper import RobotWrapper
import pybullet as p
import pinocchio as pin
import numpy as np
import time
from bullet_utils.env import BulletEnvWithGround
from Solo12 import solo12_impedance_controller
class solo12robot(object):
    def __init__(
        self,
        name, 
        pos= None,
        orn = None,


    ):








        self.name = name

        if pos is None:
            pos = [0.0, 0, .4]
            
        if orn is None:
            orn = p.getQuaternionFromEuler([0, 0, 0])


        
        self.urdf_path = Solo12Config.urdf_path
        self.meshes_path = Solo12Config.meshes_path
        self.bullet_model = p.loadURDF(
            self.urdf_path,
            pos,
            orn,
            useFixedBase= True
            )

        self.pin_model = RobotWrapper.BuildFromURDF(self.urdf_path, self.meshes_path)
        self.data = self.pin_model.data
        self.model = self.pin_model.model
        self.zerogains = np.zeros(3)
        self.bullet_joint_map = {}
        for ji in range(p.getNumJoints(self.bullet_model)):
            self.bullet_joint_map[
                p.getJointInfo(self.bullet_model, ji)[1].decode("UTF-8")
            ] = ji
        
        self.joint_names = ['FL_HAA', 'FL_HFE', 'FL_KFE', 'FR_HAA', 'FR_HFE', 'FR_KFE', 'HL_HAA', 'HL_HFE', 'HL_KFE', 'HR_HAA', 'HR_HFE', 'HR_KFE']
        self.bullet_joint_ids = np.array(
            [self.bullet_joint_map[name] for name in self.joint_names])

        
        self.pinocchio_joint_ids = np.array(
            [self.pin_model.model.getJointId(name) for name in self.joint_names]
        )
        
        vec2list = lambda m: np.array(m.T).reshape(-1).tolist()

        initial_configuration = (
        [0.2, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0]
        + 2 * [0.0, 0.8, -1.6]
        + 2 * [0.0, +0.8, -1.6]
        )   

        q0 = np.matrix(initial_configuration).T
        dq0 = np.matrix(Solo12Config.initial_velocity).T
        base_stat = p.getDynamicsInfo(self.bullet_model, -1)
        base_pos, base_quat = p.multiplyTransforms(vec2list(q0[:3]), vec2list(q0[3:7]), base_stat[3], base_stat[4])
        p.resetBasePositionAndOrientation(
            self.bullet_model, base_pos, base_quat
        )

        # Pybullet assumes the base velocity to be aligned with the world frame.
        rot = np.array(p.getMatrixFromQuaternion(q0[3:7])).reshape((3, 3))
        p.resetBaseVelocity(
            self.bullet_model, vec2list(rot.dot(dq0[:3])), vec2list(rot.dot(dq0[3:6]))
        )




        for i, bullet_joint_id in enumerate(self.bullet_joint_ids):
            p.resetJointState(
                self.bullet_model,
                bullet_joint_id,
                q0[5 + self.pinocchio_joint_ids[i]],
                dq0[4 + self.pinocchio_joint_ids[i]],
            )


        # self.bullet_joint_ids = np.array(
        #         [self.bullet_joint_map[ind] for ind in self.joint_names]
        #     )





        p.setJointMotorControlArray(
                    self.bullet_model,
                    range(16),
                    p.VELOCITY_CONTROL,
                    forces=np.zeros(16),
    )
    def send_joint_command(self, tau ):
            p.setJointMotorControlArray(
                self.bullet_model,
                range(3),
                p.TORQUE_CONTROL,
                forces=tau[0:3],
                positionGains=  self.zerogains,
                velocityGains= self.zerogains,
                )
            p.setJointMotorControlArray(
                self.bullet_model,
                range(4,7),
                p.TORQUE_CONTROL,
                forces=tau[3:6],
                positionGains=  self.zerogains,
                velocityGains= self.zerogains,
                )
            p.setJointMotorControlArray(
                self.bullet_model,
                range(8,11),
                p.TORQUE_CONTROL,
                forces=tau[6:9],
                positionGains=  self.zerogains,
                velocityGains= self.zerogains,
                )
            p.setJointMotorControlArray(
                self.bullet_model,
                range(12,15),
                p.TORQUE_CONTROL,
                forces=tau[9:12],
                positionGains=  self.zerogains,
                velocityGains= self.zerogains,
                )
        



        
env = BulletEnvWithGround()
robot = solo12robot("test")


controller = solo12_impedance_controller.RobotImpedanceController(robot)
pdes = np.ones(12)*100

ddes = np.ones(12)*20
xdes = np.array([0, 0.05,-.2, 0, -0.05,-.2, 0, 0.05, -.2, 0, -0.05, -.2])


xddes = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


fdes = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


for i in range(50000):
    
        
    all_joint_data = p.getJointStates(robot.bullet_model ,range(16))

    tau = controller.return_joint_torques( all_joint_data , pdes, ddes, xdes, xddes, fdes) 

    robot.send_joint_command(tau)

    p.stepSimulation()
    time.sleep(1./10000.)





# 