import pybullet as p
import time
import scipy

# class Dog:
p.connect(p.GUI)
plane = p.loadURDF("lib/env/a1/plane.urdf")
p.setGravity(0, 0, -9.8)
p.setTimeStep(1. / 500)
# p.setDefaultContactERP(0)
# urdfFlags = p.URDF_USE_SELF_COLLISION+p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
urdfFlags = p.URDF_USE_SELF_COLLISION
quadruped = p.loadURDF("lib/env/a1/urdf/a1.urdf", [0, 0, 0.48], [0, 0, 0, 1], flags=urdfFlags, useFixedBase=False)

# enable collision between lower legs
for j in range(p.getNumJoints(quadruped)):
    print(p.getJointInfo(quadruped, j))

lower_legs = [2, 5, 8, 11]
for l0 in lower_legs:
    for l1 in lower_legs:
        if l1 > l0:
            enableCollision = 1
            print("collision for pair", l0, l1, p.getJointInfo(quadruped, l0)[12], p.getJointInfo(quadruped, l1)[12],
                  "enabled=", enableCollision)
            p.setCollisionFilterPair(quadruped, quadruped, 2, 5, enableCollision)

jointIds = []
paramIds = []

p.enableJointForceTorqueSensor(quadruped, 5)
maxForceId = p.addUserDebugParameter("maxForce", 0, 100, 20) # this is made for debuging
p.addUserDebugLine([0, 0, 0], [1, 1, 1])

for j in range(p.getNumJoints(quadruped)):
    p.changeDynamics(quadruped, j, linearDamping=0, angularDamping=0)
    info = p.getJointInfo(quadruped, j)
    # print(info)
    jointName = info[1]
    jointType = info[2]
    if jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE:
        jointIds.append(j)

# print(jointIds)
p.getCameraImage(480, 320)
p.setRealTimeSimulation(0)

joints = []
i = 1
while (1):
    i += 1
    with open("lib/env/a1/mocap.txt", "r") as filestream:
        for line in filestream:
            maxForce = p.readUserDebugParameter(maxForceId)
            currentline = line.split(",")
            frame = currentline[0]
            t = currentline[1]
            joints = currentline[2:14]
            for j in range(12):
                targetPos = float(joints[j])  # Here is the control action
                # print(targetPos)
                p.setJointMotorControl2(quadruped, jointIds[j], p.POSITION_CONTROL, targetPos, force=maxForce)
            pos, orn = p.getBasePositionAndOrientation(quadruped)
            pos_v, orn_v = p.getBaseVelocity(quadruped)
            joint_states = p.getJointStates(quadruped, range(
                12))  # for each join it contains (position, velocity, reaction_forces(6-dim), applied torque)
            link_states = p.getLinkStates(quadruped, range(12))
            # print(link_states[5])
            # print(joint_states[5])
            p.stepSimulation()
            for lower_leg in lower_legs:
                # print("points for ", quadruped, " link: ", lower_leg)
                pts = p.getContactPoints(quadruped, -1, lower_leg)
                # print("num points=", len(pts))
                # print(pts)
                # for pt in pts:
                # print(pt)
            # print(pos, orn)
            ##### Calculating reward:
            print(i)
            time.sleep(1. / 500.)

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
