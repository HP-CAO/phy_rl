"""Gym wrapper for a1 robot"""
import math
import gym
from gym.utils import seeding
import numpy as np
import pybullet
from pybullet_utils import bullet_client as bc
import time
import os
import random

URDF_A1 = "lib/env/a1/urdf/a1.urdf"
URDF_PLANE = "lib/env/a1/plane.urdf"
URDF_Flags = pybullet.URDF_USE_SELF_COLLISION
RENDER_WIDTH = 640
RENDER_HEIGHT = 480


class A1Params:
    def __init__(self):
        self.x_threshold = 0.5
        self.if_render = False
        self.control_frequency = 10


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
        self.action_bound = 1
        self._quadruped = None
        self._plane = None
        self._timestep = 1 / self.params.control_frequency
        self._observation = []
        self._max_force = 50
        self._envStepCounter = 0
        self._jointIds = []

    def reset(self):
        self._envStepCounter = 0
        self._p.resetSimulation()
        self._p.setGravity(0, 0, -9.8)
        self._p.setTimeStep(1. / 500)
        self._quadruped = self._p.loadURDF(URDF_A1, [0, 0, 0.48], [0, 0, 0, 1], float=URDF_Flags, useFixedBase=False)
        self._plane = self._p.loadURDF(URDF_PLANE)

        lower_legs = [2, 5, 8, 11]
        for l0 in lower_legs:
            for l1 in lower_legs:
                if l1 > l0:
                    enableCollision = 1
                    self._p.setCollisionFilterPair(self._quadruped, self._quadruped, 2, 5, enableCollision)

        self._p.enableJointForceTorqueSensor(self._quadruped, 5)  # here we need to know everything for the joint

        self._jointIds = []

        for j in range(self._p.getNumJoints(self._quadruped)):
            self._p.changeDynamics(self._quadruped, j, linearDamping=0, angularDamping=0)
            info = self._p.getJointInfo(self._quadruped, j)
            jointName = info[1]
            jointType = info[2]

            if jointType == self._p.JOINT_PRISMATIC or jointType == self._p.JOINT_REVOLUTE:  # only add non-fixed joint
                self._jointIds.append(j)

        for i in range(100):
            self._p.stepSimulation()
        self._p.setTimeStep(self._timestep)

        self._observation = self.getExtendedObservation()
        return np.array(self._observation)

    def _reward(self):
        # todo here we need to implement our nice reward
        return 1

    def _performance(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def getExtendedObservation(self):
        self._observation = [0., ] * 4
        return self._observations

    def step(self, action):
        """
        Here action is a 12-dim array
        """
        if self._renders:
            basePos, orn = self._p.getBasePositionAndOrientation(self._quadruped)
            # self._p.resetDebugVisualizerCamera(1, 30, -40, basePos)
            time.sleep(self._timestep)

        for j, a in enumerate(action):
            self._p.setJointMotorControl2(self._quadruped, self._jointIds[j],
                                          self._p.POSITION_CONTROL, a, force=self._max_force)

        self._p.stepSimulation()
        self._observation = self.getExtendedObservation()
        self._envStepCounter += 1
        reward = self._reward()
        done = self._termination()

        return np.array(self._observation), reward, done, {}

    def render(self, mode='human', close=False):

        if mode != "rgb_array":
            return np.array([])

        base_pos, orn = self._p.getBasePositionAndOrientation(self._quadruped)
        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                                distance=self._cam_dist,
                                                                yaw=self._cam_yaw,
                                                                pitch=self._cam_pitch,
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
        return self._envStepCounter > 1000
