import pinocchio
from sys import argv
from os.path import dirname, join, abspath
import numpy as np
# This path refers to Pinocchio source code but you can define your own directory here.

# You should change here to set up your own URDF file or just pass it as an argument of this example.
testStand = '/home/chintu/robotsummerstuff/ws/robot_properties/robot_properties_teststand/src/robot_properties_teststand/robot_properties_teststand/pre_generated_urdf/teststand.urdf'
# Load the urdf model
model    = pinocchio.buildModelFromUrdf(testStand)
print('model name: ' + model.name)
# Create data required by the algorithms
data = model.createData()
# Sample a random configuration

q = np.array([0,np.pi/2,0])
print('q: %s' % q.T)
# Perform the forward kinematics over the kinematic tree
pinocchio.forwardKinematics(model,data,q)
# Print out the placement of each joint of the kinematic tree
for name, oMi in zip(model.names, data.oMi):
    print(("{:<24} : {: .2f} {: .2f} {: .2f}"
          .format( name, *oMi.translation.T.flat )))

