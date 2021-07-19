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




class ImpedanceControllerTest(ImpedanceController):
    def compute_impedance_torques(self, q, dq, kp, kd, x_des, xd_des, f):

        assert np.shape(x_des) == (3,)
        assert np.shape(xd_des) == (3,)
        assert np.shape(f) == (3,)
        assert np.shape(kp) == (3,)
        assert np.shape(kd) == (3,)
        

        # q = np.zeros(3)
        # dq = np.zeros(3)

        # for i in range(3):
        #     q[i] = data[i][0]
        #     dq[i] = data[i][1]

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





class RobotImpedanceController(ImpedanceController):
    def __init__(self, robot):
        """
        Input:
            robot : robot object returned by pinocchio wrapper
            config_file : file that describes the desired frames to create
                          springs in
        """

        self.robot = robot
        self.num_eef = 0
        self.imp_ctrl_array = []
        self.initialise_impedance_controllers()

    def initialise_impedance_controllers(self):
        """
        Reads the config file and initializes the impedance controllers
        Input:
            config_file : file that describes the desired frames to create
                          springs in
        """

            # TODO:
            # Check here to make sure frame names exist in pinocchio robot_model/data structure

            # Append to list of impedance controllers


        self.imp_ctrl_array.append(
            ImpedanceControllerTest(
                self.robot.name,
                self.robot.pin_model,
                "FL_HAA",
                "FL_ANKLE",
                7,
                3,
            )
        )
        self.imp_ctrl_array.append(
            ImpedanceControllerTest(
                self.robot.name,
                self.robot.pin_model,
                "FR_HAA",
                "FR_ANKLE",
                10,
                3,
            )
        )
        self.imp_ctrl_array.append(
            ImpedanceControllerTest(
                self.robot.name,
                self.robot.pin_model,
                "HL_HAA",
                "HL_ANKLE",
                13,
                3,
            )
        )
        self.imp_ctrl_array.append(
            ImpedanceControllerTest(
                self.robot.name,
                self.robot.pin_model,
                "HR_HAA",
                "HR_ANKLE",
                16,
                3,
            )
        )
    def return_joint_torques(self, all_joint_data, kp, kd, x_des, xd_des, f):
        """
        Returns the joint torques at the current timestep
        Input:
            q : current joint positions
            dq : current joint velocities
            kp : Proportional gain
            kd : derivative gain
            x_des : desired lengths with respect to the root frame for each
                    controller (3*number_of_springs)
            xd_des : desired velocities with respect to the root frame
            f : feed forward forces
        """
        
        
        for i in range(3,15,3):
            all_joint_data = np.delete(all_joint_data, i, axis = 0)

        q = np.zeros(12)
        dq = np.zeros(12)
        for k in range(12):
            q[k] = all_joint_data[k][0]
            dq[k] = all_joint_data[k][1] 


        tau = np.zeros(
            np.sum((leg.active_joints) for leg in self.imp_ctrl_array[:])
        )
        
        s1 = slice(0, 0)
        
        for k in range(len(self.imp_ctrl_array)):
            s = slice(3 * k, 3 * (k + 1))
            # print()

            s1 = slice( s1.stop, s1.stop + self.imp_ctrl_array[k].active_joints)

            # print(self.imp_ctrl_array[k].frame_end_name)
          
            # print (k, s1)
           
            # print(self.imp_ctrl_array[k].compute_impedance_torques(
            #     q, dq, kp[s1], kd[s1], x_des[s1], xd_des[s1], f[s1]
            # ))

            tau[s1] = self.imp_ctrl_array[k].compute_impedance_torques(
                q, dq, kp[s], kd[s], x_des[s], xd_des[s], f[s]
            )[s]
            # print(tau[s1])
            

        return tau
