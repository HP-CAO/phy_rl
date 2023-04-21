"""Gym wrapper for a1 robot"""
import math

import gym
from gym.utils import seeding
import numpy as np
import pybullet
from pybullet_utils import bullet_client as bc

URDF_A1 = "lib/env/a1/urdf/a1.urdf"
URDF_PLANE = "lib/env/a1/plane.urdf"
URDF_Flags = pybullet.URDF_USE_SELF_COLLISION
RENDER_WIDTH = 640
RENDER_HEIGHT = 480


class A1Params:
    def __init__(self):
        self.x_threshold = 0.5
        self.if_render = False
        self.control_frequency = 500


class A1Robot(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, params: A1Params):
        self._observations = []
        self.params = params
        self._renders = self.params.if_render

        if self._renders:
            self._p = bc.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._p = bc.BulletClient()

        self.seed()
        self.actionBound = 1  # double check with the robot
        self._quadruped = None
        self._plane = None
        self._timeStep = 1 / self.params.control_frequency
        self._observation = []
        self._maxForce = 20
        self._envStepCounter = 0
        self._jointIds = []
        self._initialize()

    def _initialize(self):
        self.reset()

    def reset(self):
        self._envStepCounter = 0
        self._p.resetSimulation()
        self._plane = self._p.loadURDF(URDF_PLANE)
        self._p.setGravity(0, 0, -9.8)
        self._p.setTimeStep(self._timeStep)
        self._quadruped = self._p.loadURDF(URDF_A1, [0, 0, 0.48], [0, 0, 0, 1], flags=URDF_Flags, useFixedBase=False)

        lower_legs = [2, 5, 8, 11]

        for l0 in lower_legs:
            for l1 in lower_legs:
                if l1 > l0:
                    enableCollision = 1
                    # print("collision for pair", l0, l1, self._p.getJointInfo(self._quadruped, l0)[12],
                    #       self._p.getJointInfo(self._quadruped, l1)[12], "enabled=", enableCollision)
                    self._p.setCollisionFilterPair(self._quadruped, self._quadruped, 2, 5, enableCollision)

        self._p.enableJointForceTorqueSensor(self._quadruped, 5)  # here we need to know everything for the joint
        self._p.addUserDebugLine([0, 0, 0], [1, 1, 1])

        self._jointIds = []

        for j in range(self._p.getNumJoints(self._quadruped)):
            self._p.changeDynamics(self._quadruped, j, linearDamping=0, angularDamping=0)
            info = self._p.getJointInfo(self._quadruped, j)
            # jointName = info[1]
            jointType = info[2]

            if jointType == self._p.JOINT_PRISMATIC or jointType == self._p.JOINT_REVOLUTE:  # only add non-fixed joint
                self._jointIds.append(j)

        self._p.getCameraImage(480, 320)
        self._p.setRealTimeSimulation(0)
        self._observation = self.getExtendedObservation()
        print("<==========   Environment has been reset   ==========>")
        return np.array(self._observation)

    def _reward(self):
        # todo here we need to implement our nice reward
        return 1

    def _performance(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def getExtendedObservation(self):
        """
        state: joint positions (12), joint velocities (12), roll pitch angles of the boby (2), foot contact indicators (4) values
        action: position control for the 12 robot joints which then use pd controller to convert it to the torques t
        """

        """
        The observation of the robot consists of: current state x_t_dim, previous_action at_1 (12);
        """
        pos, orn = self._p.getBasePositionAndOrientation(self._quadruped)  # pose of the dog
        pos_v, orn_v = self._p.getBaseVelocity(self._quadruped)  # velocity of the dog

        # observations for joints
        joint_states = self._p.getJointStates(self._quadruped, range(
            12))  # for each join it contains (position, velocity, reaction_forces(6-dim), applied torque)
        joints_states = np.array([np.concatenate(joint_states[i], axis=-1) for i in self._jointIds])

        joints_position = joints_states[:, 0]  # 12
        joints_velocity = joints_states[:, 1]  # 12
        joints_torque = joints_states[:, -1]  # 12

        contact_points = self._p.getContactPoints(self._quadruped, self._plane)
        feet_contact_states = [0, ] * 4

        if len(contact_points) != 0:
            # the order of the link index is {5, 9, 13, 17}
            for contact in contact_points:
                if contact[4] == 5:
                    feet_contact_states[0] = 1
                if contact[4] == 9:
                    feet_contact_states[1] = 1
                if contact[4] == 13:
                    feet_contact_states[2] = 1
                if contact[4] == 17:
                    feet_contact_states[3] = 1

        self._basePos = pos
        self._baseOrn = orn
        self._jointsPos = joints_position
        self._jointsVelocity = joints_velocity
        self._jointsTorque = joints_torque
        self._contact_feet = contact_points
        self._baseVelocity = pos_v
        self._baseAngularVelocity = orn_v

        self._observations = np.concatenate([joints_position, joints_velocity, pos, orn[:2], feet_contact_states])

        return self._observations

    def step(self, action):
        """
        Here action at is the desired join position for the 12 robot joints.
        """
        for j, a in enumerate(action):
            self._p.setJointMotorControl2(self._quadruped, self._jointIds[j],
                                          self._p.POSITION_CONTROL, a, force=self._maxForce)

        self._p.stepSimulation()
        self._envStepCounter += 1

        pos, orn = self._p.getBasePositionAndOrientation(self._quadruped)  # pose of the dog
        pos_v, orn_v = self._p.getBaseVelocity(self._quadruped)  # velocity of the dog

        # observations for joints
        joint_states = self._p.getJointStates(self._quadruped, range(
            12))  # for each join it contains (position, velocity, reaction_forces(6-dim), applied torque)
        joints_states = np.array([np.concatenate(joint_states[i], axis=-1) for i in self._jointIds])

        joints_position = joints_states[:, 0]  # 12
        joints_velocity = joints_states[:, 1]  # 12
        joints_torque = joints_states[:, -1]  # 12

        contact_points = self._p.getContactPoints(self._quadruped, self._plane)
        feet_contact_states = [0, ] * 4

        if len(contact_points) != 0:
            # the order of the link index is {5, 9, 13, 17}
            for contact in contact_points:
                if contact[4] == 5:
                    feet_contact_states[0] = 1
                if contact[4] == 9:
                    feet_contact_states[1] = 1
                if contact[4] == 13:
                    feet_contact_states[2] = 1
                if contact[4] == 17:
                    feet_contact_states[3] = 1

        self._basePos = pos
        self._baseOrn = orn
        self._jointsPos = joints_position
        self._jointsVelocity = joints_velocity
        self._jointsTorque = joints_torque
        self._contact_feet = contact_points
        self._baseVelocity = pos_v
        self._baseAngularVelocity = orn_v

        self._observation = self.getExtendedObservation()
        reward = self._reward()
        done = self._termination()

        return np.array(self._observation), reward, done, {}

    def render(self, mode='human', close=False):

        if mode != "rgb_array":
            return np.array([])

        base_pos, orn = self._p.getBasePositionAndOrientation(self._quadruped)
        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=(0.3, 0, 0.48),
                                                                distance=0.1,
                                                                yaw=-90,
                                                                pitch=0,
                                                                roll=0,
                                                                upAxisIndex=2)
        proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                         aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                         nearVal=0.1,
                                                         farVal=100.0)

        (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                                  height=RENDER_HEIGHT,
                                                  viewMatrix=view_matrix,
                                                  projectionMatrix=proj_matrix,
                                                  renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _termination(self):  # we need to implement the crash
        """
        One episode would terminate when the height of the body will be below 0.28m
        or the episode will exceed the maximum steps = 1000
        """
        termination = False
        if self._observations[2] < 0.28 or self._observations[3] > 0.4 or self._observations[4] > 0.2:
            termination = False

        return termination
