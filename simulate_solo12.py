from re import T
from robot_properties_solo.solo12wrapper import Solo12Config, Solo12Robot
from pinocchio.robot_wrapper import RobotWrapper
import pybullet as p
import pinocchio as pin
import numpy as np
import time
from bullet_utils.env import BulletEnvWithGround
from Solo12 import solo12_impedance_controller
import solo12robot
from matplotlib import pyplot as plt 

   
# 
env = BulletEnvWithGround()
robot = solo12robot.solo12robot("test")
controller = solo12_impedance_controller.RobotImpedanceController(robot)
robot.pin_model
pdes = np.ones(12)*500
ddes = np.ones(12)*10
xdes = np.array([0, 0.05,-.2,         0, -0.05,-.2,        0, 0.05, -.2,       0, -0.05, -.2])
xddes = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
fdes = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
impedance_data = np.array([pdes, ddes,xdes,xddes, fdes])
state = True
state2 = True
t =0
steptime= .1
pos = 0
#
xarr = []
zarr = []
xopparr = []
zopparr = []

for i in range(1000):
 
# 
    
    if t%steptime < .00001:
        if state == True:
            # pos = -steptime
            state = False
             
            
            t = 0
                        
        elif state == None:
            state = None
        else:
            # pos = +steptime
            state = True
            
            t = 0

    if t==0:
        if state2 == True:
            state2 = False
        else:
            state2 = True
    

    # if t%steptime < .00001:
    #     x = robot.compute_position()
    #     print( x[0],  "   +   ",  t+pos, "   =   ",t+pos+x[0])    

    x1 = robot.compute_position()

    # print(x1[0])

    step_data = np.array([
        -.1,
        -.1,
        1,
        .001,
        t,
        steptime,
        pos,
        xarr,
        zarr,
        xopparr,
        zopparr
        
    ])
    
    

    t,xarr,zarr,xopparr,zopparr  = robot.singlestep(controller, step_data, impedance_data, state, state2 )
    all_joint_data = p.getJointStates(robot.bullet_model ,range(16))
    tau = controller.return_joint_torques( all_joint_data , impedance_data) 
    robot.send_joint_command(tau)


    
    p.stepSimulation()
    time.sleep(1./10000.)

final = robot.mpcModel(1, .001, 100)
# print(final)
# plt.plot(xarr,zarr, 'ro')
# plt.ylabel('z values')
# plt.xlabel('t or x values')
# plt.show()
# plt.plot(xopparr,zopparr)
# plt.ylabel('z values')
# plt.xlabel('t or x values')
# plt.show()
