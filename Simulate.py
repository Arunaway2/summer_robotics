import numpy as np
import time
from bullet_utils.env import BulletEnvWithGround
import  pinocchio as pin
import pybullet as p
from teststand import TestStand_Robot
import Revised_Impedance_Controller_TestStand
import pybullet_data

# Making the enviornment
env = BulletEnvWithGround()
# Adding the robot in the enviornment, as well as initializing the robot
robot = TestStand_Robot.teststandrobot("Robot 1", [0.0, 1, 0.40])
robot2 = TestStand_Robot.teststandrobot("Robot 2",[0.0, 0, 0.40])
env.add_robot(robot)

controller = Revised_Impedance_Controller_TestStand.ImpedanceControllerTestStand(robot.name, robot.pin_model, "HFE", "END" , 2, 2 )

pdes = [1000,1000,1000]
ddes = [1,1,1]
xdes = [0,0,-.2]
xddes = [0,0,0]
fdes = [0,0,0]



for i in range(50000):
    data  = p.getJointStates(robot.bullet_model ,range(3))
    data2 = p.getJointStates(robot2.bullet_model ,range(3))
    xdes = [0,0,-.2]
    xdes[2] += .1*np.sin(i/200)
  
    tau = controller.compute_impedance_torques( data , pdes, ddes, xdes, xddes, fdes) 
    tau2 = controller.compute_impedance_torques( data2 , pdes, ddes, xdes, xddes, fdes) 
    
    
    robot.send_joint_command(tau)
    robot2.send_joint_command(tau2)

    p.stepSimulation()
    time.sleep(1./10000.)
