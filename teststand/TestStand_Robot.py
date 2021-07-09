from robot_properties_teststand.teststand_wrapper import TeststandRobot, TeststandConfig
from pinocchio.robot_wrapper import RobotWrapper
import pybullet as p
import pinocchio as pin
import numpy as np
class TestStandRobot(object):
    def __init__(
        self,
        name,
        pos,
        orn,
        fixed,
        fixed_height,
        load

    ):
        self.name = name 
        self.pos = pos
        self.orn = orn
        self.fixed = fixed 
        self.fixed_height = fixed_height
        self.load = load

########################################################################################

        urdf_path =  TeststandConfig.urdf_path
        meshes_path = TeststandConfig.meshes_path
        pin_model = RobotWrapper.BuildFromURDF(urdf_path, meshes_path)
########################################################################################

        if load:
            bullet_model = p.loadURDF(urdf_path, pos, orn,  useFixedBase = fixed)
########################################################################################
        if load:
            if fixed_height:
                p.createConstraint(
                    bullet_model,
                    0,
                    -1,
                    -1,
                    p.JOINT_FIXED,
                    [0, 0, 0],
                    [0, 0, 0.0],
                    [0, 0, fixed_height],
                )
            
            nj = p.getNumJoints(bullet_model)
            joint_names =['joint_z', 'HFE', 'KFE', 'END']
        
            bullet_joint_map = {}
            for ji in range(p.getNumJoints(robot)):
                bullet_joint_map[
                    p.getJointInfo(bullet_model, ji)[1].decode("UTF-8")
                ] = ji

            bullet_joint_ids = np.array(
                    [bullet_joint_map[ind] for ind in joint_names]
                )

            p.setJointMotorControlArray(
                        bullet_model,
                        bullet_joint_ids,
                        p.VELOCITY_CONTROL,
                        forces=np.zeros(p.getNumJoints(robot)),
        )

########################################################################################
            
            
        
########################################################################################

        