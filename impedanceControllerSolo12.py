#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pinocchio as pin
from pinocchio.utils import zero, eye
import pybullet as p
import time
import pybullet_data 
import os
from bullet_utils.env import BulletEnvWithGround
from robot_properties_solo.quadruped12wrapper import Solo12Robot
from robot_properties_solo.config import Solo12Config
from pinocchio.robot_wrapper import RobotWrapper




urdf_path =  Solo12Config.urdf_path
meshes_path = Solo12Config.meshes_path

fixed_height = True

orn = p.getQuaternionFromEuler([0, 0, 0])


p.connect(p.GUI)
p.resetSimulation()
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane = p.loadURDF("plane.urdf")
robot = p.loadURDF(urdf_path, [0,0,.5], orn,  useFixedBase = True)
p.setGravity(0, 0, -9.81)
p.setTimeStep(0.001)


if fixed_height:
        p.createConstraint(
            robot,
            0,
            -1,
            -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, 0.0],
            [0, 0, fixed_height],
        )

class ImpedanceController(object):
    def __init__(
        self,
        name, 
        pin_robot, 
        frame_root_name, 
        frame_end_name, 
        start_colomn, 
        active_joints
    ):
        
        self.name = name
        self.pin_robot = pin_robot
        self.frame_root_name = frame_root_name
        self.frame_end_name = frame_end_name 
        self.frame_root_idx = self.pin_robot.model.getFrameId(
            self.frame_root_name)
        self.frame_end_idx = self.pin_robot.model.getFrameId(
            self.frame_end_name)
        self.start_colomn = start_colomn
        self.active_joints = active_joints

    def compute_forward_kinematics(self, q):
#         print(self.pin_robot.model, self.pin_robot.data, q)
        
        pin.framesForwardKinematics(
            
            self.pin_robot.model, self.pin_robot.data, q
        ) 

    def compute_distance_between_frames(self, q):
        return ( self.pin_robot.data.oMf[self.frame_end_idx].translation- self.pin_robot.data.oMf[self.frame_root_idx].translation)
    

    def compute_relative_velocity_between_frames(self,q, dq):
        frame_config_root = pin.SE3( self.pin_robot.data.oMf[self.frame_root_idx].rotation, np.zeros((3,1)),
        )
        frame_config_end = pin.SE3(self.pin_robot.data.oMf[self.frame_end_idx].rotation, np.zeros((3,1)),
        )
       
        vel_root_in_world_frame = frame_config_root.action.dot(
            pin.computeFrameJacobian(
                self.pin_robot.model,
                self.pin_robot.data,
                q,
                self.frame_root_idx,
            )
        ).dot(dq)[0:3]
        vel_end_in_world_frame = frame_config_end.action.dot(
            pin.computeFrameJacobian(
                self.pin_robot.model,
                self.pin_robot.data,
                q,
                self.frame_end_idx,
            )   
        ).dot(dq)[0:3]
        
        return np.subtract(vel_end_in_world_frame, vel_root_in_world_frame).T
        

    def compute_jacobian(self, q):
        self.compute_forward_kinematics(q)
        jac = pin.computeFrameJacobian(
            self.pin_robot.model, self.pin_robot.data, q, self.frame_end_idx
            
        )
        jac = self.pin_robot.data.oMf[self.frame_end_idx].rotation.dot(
            jac[0:3]
        )
        return jac
  
    def compute_impedance_torques(self, q, dq, kp, kd, x_des, xd_des, f):
        assert np.shape(x_des) == (3,)
        assert np.shape(xd_des) == (3,)
        assert np.shape(f) == (3,)
        assert np.shape(kp) == (3,)
        assert np.shape(kd) == (3,)
        
        x_des = np.array(x_des)
        xd_des = np.array(xd_des)
        f = np.array(f)
        kp = np.array([[kp[0], 0, 0], [0, kp[1], 0], [0, 0, kp[2]]])

        kd = np.array([[kd[0], 0, 0], [0, kd[1], 0], [0, 0, kd[2]]])
        
        ###########################################################


        self.compute_forward_kinematics(q)
        x = self.compute_distance_between_frames(q)
        xd = self.compute_relative_velocity_between_frames(q, dq)
        jac = self.compute_jacobian(q)[
            :,
            self.start_colomn : self.start_colomn + len(self.active_joints),
        ]
        jac = jac[:, self.active_joints]
        
        
      
        
        self.F_ = ( 
            
            f + np.matmul(kp,(x - x_des)) + np.matmul(kd, (xd - xd_des).T).T 
            
                  )
        tau = -jac.T.dot(self.F_.T)
    
        
    
        
        final_tau = []
        j = 0
        for i in range(len(self.active_joints)):
            if self.active_joints[i] == False:
                final_tau.append(0)
            else:
                final_tau.append(tau[j])
                j += 1 
        return final_tau
    
    def compute_impedance_torques_world(self, q, dq, kp, kd, x_des, xd_des, f):
        assert np.shape(x_des) == (3,)
        assert np.shape(xd_des) == (3,)
        assert np.shape(f) == (3,)
        assert np.shape(kp) == (3,)
        assert np.shape(kd) == (3,)
        
        x_des = np.array(x_des)
        xd_es = np.array(xd_des)
        f = np.array(f)
        kp = np.array(kp)
        kd = np.array(kd)
        
        self.compute_forward_kinematics(q)
        jac = self.compute_jacobian(q)
        
        x = self.pin_robot.data.oMf[self.frame_end_idx].translation
        xd = jac.dot(dq)
        
        jac = jac[
            :,
            self.start_colomn : self.start_colomn +len(self.active_joints),
        ]
        jac = jac[:, self.active_joint[i]]
        
        self.F_ = (
            f + np.matmul(kp, (x - x_des)) + np.matmul(kd, (xd - xd_des).T).T
        )
        tau = -jac.T.dot(self.F_.T)
        final_tau = []
        j = 0
        for i in range (len(self.active_joints)):
            if self.active_joints[i] == False:
                final_tau.append(0)
            else:
                final_tau.append(tau[j])
                j += 1 
        return final_tau    




class ImpedanceSolo12(ImpedanceController):
    def compute_impedance_torques(self, q, dq, kp, kd, x_des, xd_des, f):
        assert np.shape(x_des) == (3,)
        assert np.shape(xd_des) == (3,)
        assert np.shape(f) == (3,)
        assert np.shape(kp) == (3,)
        assert np.shape(kd) == (3,)
        
        
        x_des = np.array(x_des)
        xd_des = np.array(xd_des)
        f = np.array(f)
        kp = np.array(kp)
        kd = np.array(kd)
        
        self.compute_forward_kinematics(q)
        x = self.compute_distance_between_frames(q)
        xd = self.compute_relative_velocity_between_frames(q,dq)
        
        jac = self.compute_jacobian(q)
        
        self.F_ = f + kp *(x - x_des) + kd * (xd - xd_des)
        # print(kp)


        # print( x)
        # print("######################################")
        # print( x_des)
        # print("######################################")
        # # print(jac)
        
        tau = -jac.T.dot(self.F_)
        return tau            

    def compute_impedance_torques_world(self, q, dq, kp, kd, x_des, xd_des, f):
    
        assert np.shape(x_des) == (3,)
        assert np.shape(xd_des) == (3,)
        assert np.shape(f) == (3,)
        assert np.shape(kp) == (3,)
        assert np.shape(kd) == (3,)
        
        
        x_des = np.array(x_des)
        xd_des = np.array(xd_des)
        f = np.array(f)
        kp = np.array(kp)
        kd = np.array(kd)
        
        
        self.compute_forward_kinematics(q)
        jac = self.compute_jacobian(q)
        
        x = self.pin_robot.data.oMf[self.frame_end_idx].translation
        xd = jac.dot(dq)
     
        self.F_ = f + kp * (x - x_des) + kd * (xd - xd_des)
        tau = -jac.T.dot(self.F_)
        
        return tau 





robot_model = RobotWrapper.BuildFromURDF(urdf_path, meshes_path)
pin.framesForwardKinematics(robot_model.model, robot_model.data, np.zeros(3))
print(robot)
testController = ImpedanceControllerTestStand("TestStand", robot_model, "joint_z", "END", 2, 2 )
# useFixedBase=True
nj = p.getNumJoints(robot)

joint_names =['joint_z', 'HFE', 'KFE', 'END']


def move_joints(self, tau):
    p.setJointMotorControlArray(
        self.robot_id,
        self.bullet_joint_ids,
        p.TORQUE_CONTROL,
        forces=tau[self.pin2bullet_joint_only_array],
        positionGains=zeroGains,
        velocityGains=zeroGains,
    )

# print(p.getJointInfo(robot))








# print(joint_names)





bullet_joint_map = {}
for ji in range(p.getNumJoints(robot)):
    bullet_joint_map[
        p.getJointInfo(robot, ji)[1].decode("UTF-8")
    ] = ji

bullet_utils.env import BulletEnvWithGround
from robot_properties_teststand.teststand_wrapper
# print(bullet_joint_map)


bullet_joint_ids = np.array(
        [bullet_joint_map[name] for name in joint_names]
    )

# print(bullet_joint_ids)




pinocchio_joint_ids = np.array(
    [robot_model.model.getJointId(name) for name in joint_names]
)

print(pinocchio_joint_ids)

pin2bullet_joint_only_array = []

# if not useFixedBase:
#             for i in range(2, nj + 2):
#                 pin2bullet_joint_only_array.append(
#                     np.where(pinocchio_joint_ids == i)[0][0]
#                 )
# else:
#             for i in range(1, nj + 1):
#                 pin2bullet_joint_only_array.append(
#                     np.where(pinocchio_joint_ids == i)[0][0]
#                 )

print(pin2bullet_joint_only_array)





p.setJointMotorControlArray(
            robot,
            bullet_joint_ids,
            p.VELOCITY_CONTROL,
            forces=np.zeros(p.getNumJoints(robot)),
        )








for i in range(500000):
        # TODO: Implement a controller here.
        data  = p.getJointStates(robot,range(1,3))

        print(data)
        q = np.zeros(2)
        v = np.zeros(2)
        for i in range(2):
            q[i] = data[i][0]
            v[i] = data[i][1]




        pdes = [50,50,50]
        ddes = [1,1,1]
        xdes = [0,0,-.2]
        xddes = [0,0,0]
        fdes = [0,0,0]
        tau = np.zeros(4)
        taus = testController.compute_impedance_torques( q, v, pdes , ddes, xdes, xddes, fdes)
        tau =np.concatenate((tau,taus), axis = None, out = None)
        
        # print("BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB")
        # print (taus)
        # print("EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
        
        # tau[0]= 0
        # tau[1]= 0
        # tau[2] =.1
        # tau[3] =0
        print(tau)
        zeroGains = tau.shape[0] * (0.0,)

        
        
        p.setJointMotorControlArray(
            robot,
            bullet_joint_ids,
            p.TORQUE_CONTROL,
            forces=tau,
            positionGains=  zeroGains,
            velocityGains= zeroGains,
            )

        
        # Step the simulator.
        p.stepSimulation()
        time.sleep(1./10000.)

