import random

import pybullet as p
import time
import scipy
import math
import numpy as np
from PIL import Image

# class Dog:
p.connect(p.GUI)
p.setGravity(0, 0, -9.8)

# marble = p.loadURDF("lib/env/a1/marble_cube.urdf", [1, 2, 0.5], [0, 0, 0, 1])
# p.setTimeStep(1. / 500)
# p.setDefaultContactERP(0)
# urdfFlags = p.URDF_USE_SELF_COLLISION+p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS

urdfFlags = p.URDF_USE_SELF_COLLISION
q_1 = p.getQuaternionFromEuler([0, 0, math.pi])

p.getCameraImage(480, 320)
p.setRealTimeSimulation(0)

joints = []
i = 0
while 1:
    p.resetSimulation()
    theta_delta_x = np.random.uniform(-0.2, 0.2)
    theta_delta_y = np.random.uniform(-0.2, 0.2)
    theta_delta_z = np.random.uniform(-0.2, 0.2)
    q = p.getQuaternionFromEuler([0.5 * math.pi + theta_delta_x, theta_delta_y, 0.5 * math.pi + theta_delta_z])

    plane = p.loadURDF("lib/env/a1/plane.urdf")

    x = np.random.uniform(-3.5, -1.2)
    y = np.random.uniform(-1, 1)
    z = np.random.uniform(0.1, 0.3)

    stopsign = p.loadURDF("lib/env/a1/stopsign.urdf", [x, y, z], q)
    view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=(0.0, 0, 1.3),
                                                      distance=0.1,
                                                      yaw=90,
                                                      pitch=0,
                                                      roll=0,
                                                      upAxisIndex=2)
    proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                               aspect=float(640) / 480,
                                               nearVal=0.1,
                                               farVal=100.0)

    (_, _, px, _, seg) = p.getCameraImage(width=640,
                                          height=480,
                                          viewMatrix=view_matrix,
                                          projectionMatrix=proj_matrix,
                                          renderer=p.ER_BULLET_HARDWARE_OPENGL)

    rgb_array = np.array(px)
    rgb_array = rgb_array[:, :, :3]
    im = Image.fromarray(rgb_array)
    im.save(f"data/images/rgb_{i}.png")

    seg_array = np.array(seg, dtype=np.uint8) * 255
    seg_array[seg_array <= 0] = 0
    seg_array = np.stack((seg_array,) * 3, axis=-1)
    im_seg = Image.fromarray(seg_array)
    im_seg.save(f"data/labels/segmentations_{i}.png")

    i += 1
    if i == 200:
        break

# for j in range (p.getNumJoints(quadruped)):
#     p.changeDynamics(quadruped,j,linearDamping=0, angularDamping=0)
#     info = p.getJointInfo(quadruped,j)
#     js = p.getJointState(quadruped,j)
#     #print(info)
#     jointName = info[1]
#     jointType = info[2]
#     if (jointType==p.JOINT_PRISMATIC or jointType==p.JOINT_REVOLUTE):
#             paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"),-4,4,(js[0]-jointOffsets[j])/jointDirections[j]))


# p.setRealTimeSimulation(1)

# while (1):
#     for i in range(len(paramIds)):
#         c = paramIds[i]
#         targetPos = p.readUserDebugParameter(c)
#         maxForce = p.readUserDebugParameter(maxForceId)
#         p.setJointMotorControl2(quadruped,jointIds[i],p.POSITION_CONTROL,jointDirections[i]*targetPos+jointOffsets[i], force=maxForce)
