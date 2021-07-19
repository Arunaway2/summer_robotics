from robot_properties_teststand.teststand_wrapper import TeststandRobot, TeststandConfig
from pinocchio.robot_wrapper import RobotWrapper
import pybullet as p
import pinocchio as pin
import numpy as np
class teststandrobot(object):
    def __init__(
        self,
        name, 
        pos,
        orn = None,


    ):
        self.name = name

        if pos is None:
            pos = [0.0, 0, 0.40]
            
        if orn is None:
            orn = p.getQuaternionFromEuler([0, 0, 0])


        
        self.urdf_path = TeststandConfig.urdf_path
        self.meshes_path = TeststandConfig.meshes_path
        self.bullet_model = p.loadURDF(
            self.urdf_path,
            pos,
            orn,
            useFixedBase= False,
        )
        self.pin_model = RobotWrapper.BuildFromURDF(self.urdf_path, self.meshes_path)
        self.nj = p.getNumJoints(self.bullet_model)
        self.joint_names =['joint_z', 'HFE', 'KFE', 'END']
        self.data = self.pin_model.data
        self.model = self.pin_model.model
        self.zerogains = np.zeros(2)

        
        
        bullet_joint_map = {}
        for ji in range(p.getNumJoints(self.bullet_model)):
            bullet_joint_map[
                p.getJointInfo(self.bullet_model, ji)[1].decode("UTF-8")
            ] = ji

        bullet_joint_ids = np.array(
                [bullet_joint_map[ind] for ind in self.joint_names]
            )

        p.setJointMotorControlArray(
                    self.bullet_model,
                    bullet_joint_ids,
                    p.VELOCITY_CONTROL,
                    forces=np.zeros(p.getNumJoints(self.bullet_model)),
    )

    def send_joint_command(self, tau):
        p.setJointMotorControlArray(
            self.bullet_model,
            range (1,3),
            p.TORQUE_CONTROL,
            forces=tau[1:],
            positionGains=  self.zerogains,
            velocityGains= self.zerogains,
            )
