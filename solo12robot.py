from re import T
from robot_properties_solo.solo12wrapper import Solo12Config, Solo12Robot
from pinocchio.robot_wrapper import RobotWrapper
import pybullet as p
import pinocchio as pin
import numpy as np
import time
from bullet_utils.env import BulletEnvWithGround
from Solo12 import solo12_impedance_controller
from matplotlib import pyplot as plt 
class solo12robot(object):
    def __init__(
        self,
        name, 
        pos= None,
        orn = None,


    ):
        self.name = name

        if pos is None:
            pos = [0.0, 0, .25]
            
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
        [0.2, 0.0, 1, 0.0, 0.0, 0.0, 1.0]
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






    def singlestep( self, controller, step_data, impedance_data, state, state2):


        # length, height, fraction , delta_sim_step, current_sim_step, iteration 
        if state == None:

            
            all_joint_data = p.getJointStates(self.bullet_model ,range(16))
            tau = controller.return_joint_torques( all_joint_data , impedance_data) 
            self.send_joint_command(tau)

            step = 0 
            return step


        elif state == True:

            length = step_data[0] 
            height = step_data[1] 
            fraction = step_data[2] 
            delta_sim_step = step_data[3] 
            current_sim_step = step_data[4]
            steptime = step_data[5]
            pos = step_data[6]
            xarr = step_data[7]
            zarr = step_data[8]
            xopparr = step_data[9]
            zopparr = step_data[10]
            
            

            current_sim_step += delta_sim_step
            # print(current_sim_step)
            # print(pos)
            x = current_sim_step 
            x_opp = -current_sim_step 

            z =  height*np.sin((np.pi/(-1*steptime))*x)
            

            # z =  -.2-height*np.sin((np.pi/length)*x)

            impedance_data[2][0] = x
            
            impedance_data[2][9] = x
            temp = impedance_data[2][9]
            xarr.append(temp) 


            # x positions for the 2 stepping legs

            impedance_data[2][2] = z-.2
            impedance_data[2][11] = z-.2
            temp2 = impedance_data[2][11]
            
            zarr.append(temp2+.2)
            # z positions for the 2 stepping legs


            impedance_data[2][3] = x_opp
            impedance_data[2][6] = x_opp
            # x positions for the 2 pivoting legs


           
            


            all_joint_data = p.getJointStates(self.bullet_model ,range(16))
            tau = controller.return_joint_torques( all_joint_data , impedance_data) 
            self.send_joint_command(tau)
            

            return current_sim_step, xarr,zarr, xopparr, zopparr

        elif state == False:

              
            length = step_data[0] 
            height = step_data[1] 
            fraction = step_data[2] 
            delta_sim_step = step_data[3] 
            current_sim_step = step_data[4]
            steptime = step_data[5]
            pos = step_data[6]
            xarr = step_data[7]
            zarr = step_data[8]
            xopparr = step_data[9]
            zopparr = step_data[10]

            current_sim_step += delta_sim_step
            
            x = current_sim_step - steptime
            x_opp = -current_sim_step + steptime
 
            z =  -.2-height*np.sin((np.pi/(-1*steptime))*x)

            # z =  -.2-height*np.sin((np.pi/length)*x)

            impedance_data[2][3] = x
            impedance_data[2][6] = x
            # x positions for the 2 stepping legs


            impedance_data[2][5] = z
            impedance_data[2][8] = z
            # z positions for the 2 stepping legs


            impedance_data[2][0] = x_opp
            impedance_data[2][9] = x_opp

            temp = impedance_data[2][9]
            # print(temp)
            

            xopparr.append(temp)
            zopparr.append(0)
            # x positions for the 2 pivoting legs

            
            all_joint_data = p.getJointStates(self.bullet_model ,range(16))
            tau = controller.return_joint_torques( all_joint_data , impedance_data) 
            self.send_joint_command(tau)

            return current_sim_step, xarr, zarr, xopparr,zopparr

    def compute_position(self):
        all_joint_data = p.getJointStates(self.bullet_model ,range(16))
        q1 = np.zeros(12)
        q2 = np.zeros(12)
        q3 = np.zeros(12)
        q4 = np.zeros(12)
        for i in range(3):
            q1[i] = all_joint_data[i][0]
        # for j in range(4,7):
        #     q2[j] = all_joint_data[j][0]
        # for k in range(8,11):
        #     q3[k] = all_joint_data[k][0]

        # for l in range(12,15):
        #     q4[l] = all_joint_data[l][0]
        

        

        pin.framesForwardKinematics(
                
                self.pin_model.model, self.pin_model.data, q1
            ) 
        x1 = ( self.pin_model.data.oMf[self.pin_model.model.getFrameId("FL_ANKLE")].translation - self.pin_model.data.oMf[self.pin_model.model.getFrameId("FL_HAA")].translation)




        # pin.framesForwardKinematics(
                
        #         self.pin_model.model, self.pin_model.data, q2
        #     ) 
        # x2 = ( self.pin_model.data.oMf[self.pin_model.model.getFrameId("FR_ANKLE")].translation - self.pin_model.data.oMf[self.pin_model.model.getFrameId("FR_HAA")].translation)


        # pin.framesForwardKinematics(
                
        #         self.pin_model.model, self.pin_model.data, q3
        #     ) 
        # x3 = ( self.pin_model.data.oMf[self.pin_model.model.getFrameId("HL_ANKLE")].translation - self.pin_model.data.oMf[self.pin_model.model.getFrameId("HL_HAA")].translation)


        # pin.framesForwardKinematics(
                
        #         self.pin_model.model, self.pin_model.data, q4
        #     ) 
        # x4 = ( self.pin_model.data.oMf[self.pin_model.model.getFrameId("HR_ANKLE")].translation - self.pin_model.data.oMf[self.pin_model.model.getFrameId("HR_HAA")].translation)



        


        return x1


    def mpcModel(self, m, dt, size):
        s = np.ones(size)
        a = np.zeros(shape=(size, size))
        s1 = np.array([1,dt,0,-1,0])
        s2 = np.array([0,1,dt/m,0,-1])
        np.fill_diagonal(a, s1)
        print( a )

        
            




        

        # print(a)


        # a = np.array([[1,dt/mass,0,-1,0,0,0,0,0,0,],
        #               [0,1,dt/mass,0,-1,0,0,0,0,0]])

        return a.dot(s)

        # print(a*s)



