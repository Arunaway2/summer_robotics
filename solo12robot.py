from re import T
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






    def singlestep( self, step_data, impedance_data, state):

#
        # length, height, fraction , delta_sim_step, current_sim_step, iteration 
        if state == None:

            
            all_joint_data = p.getJointStates(robot.bullet_model ,range(16))
            tau = controller.return_joint_torques( all_joint_data , impedance_data) 
            robot.send_joint_command(tau)

            step = 0 
            return step


        elif state == True:
             
            length = step_data[0] 
            height = step_data[1] 
            fraction = step_data[2] 
            delta_sim_step = step_data[3] 
            current_sim_step = step_data[4]
            steptime = step_data[5]

            current_sim_step -= delta_sim_step
            
            x = current_sim_step
            x_opp = -current_sim_step 

            z =  -.2-height*np.sin((np.pi/(-1*steptime))*current_sim_step)

            # z =  -.2-height*np.sin((np.pi/length)*x)

            impedance_data[2][0] = x
            impedance_data[2][9] = x
            # x positions for the 2 stepping legs


            impedance_data[2][2] = z
            impedance_data[2][11] = z
            # z positions for the 2 stepping legs


            impedance_data[2][3] = x_opp
            impedance_data[2][6] = x_opp
            # x positions for the 2 pivoting legs

            
            
            all_joint_data = p.getJointStates(robot.bullet_model ,range(16))
            tau = controller.return_joint_torques( all_joint_data , impedance_data) 
            robot.send_joint_command(tau)

            return current_sim_step

        elif state == False:

              
            length = step_data[0] 
            height = step_data[1] 
            fraction = step_data[2] 
            delta_sim_step = step_data[3] 
            current_sim_step = step_data[4]
            steptime = step_data[5]

            current_sim_step -= delta_sim_step
            
            x = current_sim_step
            x_opp = -current_sim_step 

            z =  -.2-height*np.sin((np.pi/(-1*steptime))*current_sim_step)

            # z =  -.2-height*np.sin((np.pi/length)*x)

            impedance_data[2][3] = x
            impedance_data[2][6] = x
            # x positions for the 2 stepping legs


            impedance_data[2][5] = z
            impedance_data[2][8] = z
            # z positions for the 2 stepping legs


            impedance_data[2][0] = x_opp
            impedance_data[2][9] = x_opp
            # x positions for the 2 pivoting legs

            
            
            all_joint_data = p.getJointStates(robot.bullet_model ,range(16))
            tau = controller.return_joint_torques( all_joint_data , impedance_data) 
            robot.send_joint_command(tau)

            return current_sim_step


    def compute_position(self):
        all_joint_data = p.getJointStates(robot.bullet_model ,range(16))
        q1 = np.zeros(12)
        q2 = np.zeros(12)
        q3 = np.zeros(12)
        q4 = np.zeros(12)
        

        for i in range(3):
            q1[i] = all_joint_data[i][0]
        for i in range(3,6):
            q2[i] = all_joint_data[i][0]
        for i in range(6,9):
            q3[i] = all_joint_data[i][0]
        for i in range(9,12):
            q4[i] = all_joint_data[i][0]
        
        

        pin.framesForwardKinematics(
                
                robot.pin_model.model, robot.pin_model.data, q1
            ) 
        x1 = ( robot.pin_model.data.oMf[robot.pin_model.model.getFrameId("FL_ANKLE")].translation - robot.pin_model.data.oMf[robot.pin_model.model.getFrameId("FL_HAA")].translation)




        pin.framesForwardKinematics(
                
                robot.pin_model.model, robot.pin_model.data, q2
            ) 
        x2 = ( robot.pin_model.data.oMf[robot.pin_model.model.getFrameId("FR_ANKLE")].translation - robot.pin_model.data.oMf[robot.pin_model.model.getFrameId("FR_HAA")].translation)


        pin.framesForwardKinematics(
                
                robot.pin_model.model, robot.pin_model.data, q3
            ) 
        x3 = ( robot.pin_model.data.oMf[robot.pin_model.model.getFrameId("HL_ANKLE")].translation - robot.pin_model.data.oMf[robot.pin_model.model.getFrameId("HL_HAA")].translation)


        pin.framesForwardKinematics(
                
                robot.pin_model.model, robot.pin_model.data, q4
            ) 
        x4 = ( robot.pin_model.data.oMf[robot.pin_model.model.getFrameId("HR_ANKLE")].translation - robot.pin_model.data.oMf[robot.pin_model.model.getFrameId("HR_HAA")].translation)






        return x1,x2,x3,x4

        
env = BulletEnvWithGround()
robot = solo12robot("test")


controller = solo12_impedance_controller.RobotImpedanceController(robot)
robot.pin_model
#
pdes = np.ones(12)*500
ddes = np.ones(12)*10
# xdes = np.zeros(12)
xdes = np.array([0, 0.05,-.2,         0, -0.05,-.2,        0, 0.05, -.2,       0, -0.05, -.2])
xddes = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
fdes = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

impedance_data = np.array([pdes, ddes,xdes,xddes, fdes])

#################################################

    # counter = -.2
        # length = -.2
        # height = -.07
        # if step < counter:
        #     step = counter
        # else:
        #     step -= .0001
            # x = step
        # z =  -.2-height*np.sin((np.pi/length)*x)


state = True
# state2 = True
t =0

steptime= .2

for i in range(50000):
    
 

    

    



   


    # print()

 
#
    
    if t%steptime < .00001:
        t = 0
        
        if state == True:
            state = False
            
        elif state == None:
            state = None
        else:
            state = True
    

    
    
    step_data = np.array([
        -.06,
        -.03,
        1,
        .0001,
        t,
        steptime
    ])

    x1,x2,x3,x4 = robot.compute_position()

    # print(x1[0], "          ", x2[0],"          ", x3[0],"          ", x4[0])
    

    # # length, height, fraction , delta_sim_step, current_sim_step, step time

    

    t = robot.singlestep(step_data, impedance_data, state)
    # # print(t)
    
    p.stepSimulation()
    time.sleep(1./10000.)



#    
    #     print(x, z, step)
        
    #     xdes[0] = x
    #     xdes[9] = x
    #  # x positions for the 2 stepping legs


    #     xdes[2] = z
    #     xdes[11] = z
    #  # z positions for the 2 stepping legs


    #     xdes[3] = x_opp
    #     xdes[6] = x_opp
    #  # x positions for the 2 pivoting legs

    #     all_joint_data = p.getJointStates(robot.bullet_model ,range(16))
    #     tau = controller.return_joint_torques( all_joint_data , impedance_data) 
    #     robot.send_joint_command(tau)

   





# 