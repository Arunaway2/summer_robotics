#!/usr/bin/env python
# coding: utf-8



import numpy as np
from numpy.core.numeric import True_
import pinocchio as pin
from pinocchio.utils import zero, eye
import pybullet as p
import time
import pybullet_data 
import os
from bullet_utils.env import BulletEnvWithGround
from robot_properties_teststand.teststand_wrapper import TeststandRobot, TeststandConfig
from pinocchio.robot_wrapper import RobotWrapper
from teststand import TestStand_Robot

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
  
    def compute_impedance_torques(self, data, kp, kd, x_des, xd_des, f):
        assert np.shape(x_des) == (3,)
        assert np.shape(xd_des) == (3,)
        assert np.shape(f) == (3,)
        assert np.shape(kp) == (3,)
        assert np.shape(kd) == (3,)

        q = np.zeros(3)
        dq = np.zeros(3)

        for i in range(3):
            q[i] = data[i][0]
            dq[i] = data[i][1]
        
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
        jac = jac[:, self.active_joints]
        
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




class ImpedanceControllerTestStand(ImpedanceController):
    def compute_impedance_torques(self, data, kp, kd, x_des, xd_des, f):

        assert np.shape(x_des) == (3,)
        assert np.shape(xd_des) == (3,)
        assert np.shape(f) == (3,)
        assert np.shape(kp) == (3,)
        assert np.shape(kd) == (3,)
        

        q = np.zeros(3)
        dq = np.zeros(3)

        for i in range(3):
            q[i] = data[i][0]
            dq[i] = data[i][1]

        print(type(q))
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




