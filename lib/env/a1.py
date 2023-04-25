"""Gym wrapper for a1 robot"""
import copy
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
        self.control_frequency = 100
        self.observe_reference_states = False
        self.random_reset_eval = False
        self.random_reset_train = False
        self.action_bound = 1.5


class A1Robot(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, params: A1Params):
        self._observations = []
        self.params = params
        self._renders = self.params.if_render
        self.seed()
        self.actionBound = 1  # double check with the robot
        self.action_dim = 12
        self._quadruped = None
        self._plane = None
        self._timeStep = 1 / self.params.control_frequency
        self._observation = []
        self._maxForce = 20
        self._envStepCounter = 0
        self._jointIds = []

        self._basePos = []
        self._baseOrn = []
        self._jointsPos = []
        self._jointsPosPre = []
        self._jointsVelocity = []
        self._jointsTorque = []
        self._jointsTorquePre = []

        self._contact_feet = []
        self._baseVelocity = []
        self._baseAngularVelocity = []
        self._observationPre = []
        self._observation = []
        self._actions = []
        self._feetVelocity = []
        self._feetForce = []
        self._feetForcePre = []
        self._feetLinkId = [5, 9, 13, 17]
        self._p = None

        self._initialize()
        self.states_observations_refer_dim = 0  # todo check when we use the reference model

    def states_observations_dim(self):
        return len(self.getExtendedObservation())

    def _initialize(self):
        self.reset()

    def random_reset(self):
        pass

    def reset(self, reset_states=None, vis=False):
        if self._renders or vis:
            self._p = bc.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._p = bc.BulletClient()

        self._envStepCounter = 0
        self._p.resetSimulation()
        self._plane = self._p.loadURDF(URDF_PLANE)
        self._p.setGravity(0, 0, -9.8)
        self._p.setTimeStep(self._timeStep)
        self._quadruped = self._p.loadURDF(URDF_A1, [-3, 0, 0.40], [0, 0, 0, 1], flags=URDF_Flags, useFixedBase=False)

        lower_legs = [2, 5, 8, 11]

        for l0 in lower_legs:
            for l1 in lower_legs:
                if l1 > l0:
                    enableCollision = 1
                    self._p.setCollisionFilterPair(self._quadruped, self._quadruped, 2, 5, enableCollision)

        self._p.addUserDebugLine([0, 0, 0], [1, 1, 1])

        self._jointIds = []

        for j in range(self._p.getNumJoints(self._quadruped)):
            self._p.changeDynamics(self._quadruped, j, linearDamping=0, angularDamping=0)
            info = self._p.getJointInfo(self._quadruped, j)
            # jointName = info[1]
            jointType = info[2]

            if jointType == self._p.JOINT_PRISMATIC or jointType == self._p.JOINT_REVOLUTE:  # only add non-fixed joint
                self._jointIds.append(j)

        for footId in self._feetLinkId:
            self._p.enableJointForceTorqueSensor(self._quadruped, footId)

        self._p.getCameraImage(480, 320)
        self._p.setRealTimeSimulation(0)

        self._update_states()
        stable_motion = np.array([0.037199, 0.660252, -1.200187, -0.028954, 0.618814, -1.183148,
                                  0.048225, 0.690008, -1.254787, -0.050525, 0.661355, -1.243304])

        for _ in range(100):
            self.step(stable_motion)

        print("<==========   Environment has been reset   ==========>")
        return np.array(self._observation)

    def _reward(self):

        # positional reward to encourage moving toward original
        # r1 = -1 * (pow(self._basePos[0], 2)) * 20
        r1 = -1 * min(self._baseVelocity[0], 0.35) * 20

        # penalize the lateral movement and rotation
        r2 = -1 * (pow(self._baseAngularVelocity[2], 2) + pow(self._baseVelocity[1], 2)) * 21

        # penalize the work
        r3 = -1 * (abs(np.array(self._jointsTorque) @ np.subtract(self._jointsTorque, self._jointsTorquePre).T)) * 0.002

        # penalize the ground impact
        r4 = -1 * (np.linalg.norm(np.subtract(self._feetForce, self._feetForcePre).flatten())) * 0.02

        # penalize the jerky motions
        r5 = -1 * (np.linalg.norm(np.subtract(self._jointsTorque, self._jointsTorquePre))) * 0.001

        # penalize the actions
        r6 = -1 * (np.linalg.norm(self._actions)) * 0.07

        # penalize the joint speeds
        r7 = -1 * (np.linalg.norm(self._jointsVelocity)) * 0.002

        # penalize the orientations
        r8 = -1 * (pow(self._baseOrn[0], 2) + pow(self._baseOrn[1], 2)) * 1.5

        # penalize the z acceleration
        r9 = -1 * (pow(self._baseVelocity[2], 2)) * 2.0

        # penalize the foot slip
        r10 = -1 * (np.linalg.norm(np.matmul(np.diag(self._contact_feet), self._feetVelocity)).flatten()) * 0.8

        reward = r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9 + r10

        return reward

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

        self._observations = np.concatenate([self._jointsPos, self._jointsVelocity,
                                             self._basePos, self._baseOrn[:2], self._contact_feet])

        return self._observations

    def _update_states(self):

        self._jointsPosPre = self._jointsPos
        self._jointsTorquePre = self._jointsTorque
        self._feetForcePre = self._feetForce

        pos, _ = self._p.getBasePositionAndOrientation(self._quadruped)  # pose of the dog in the world coordinate

        # get the rotation angles from the body frame
        _, _, _, qua_orn_local, _, _ = self._p.getLinkState(self._quadruped, 0)
        orn = self._p.getEulerFromQuaternion(qua_orn_local)

        # velocity of the dog in the world coordinate, the angular velocity should be in the body frame
        pos_v, orn_v = self._p.getBaseVelocity(self._quadruped)

        # observations for joints
        # for each join it contains (position, velocity, reaction_forces(6-dim), applied torque)
        joint_states = self._p.getJointStates(self._quadruped, range(self._p.getNumJoints(self._quadruped)))

        joints_position = []
        joints_velocity = []
        joints_torque = []

        for i in self._jointIds:
            joints_position.append(joint_states[i][0])
            joints_velocity.append(joint_states[i][1])
            joints_torque.append(joint_states[i][-1])

        contact_points = self._p.getContactPoints(self._quadruped, self._plane)
        feet_contact_states = [0, ] * 4

        if len(contact_points) != 0:
            # the order of the link index is {5, 9, 13, 17}
            for contact in contact_points:
                for k, linkId in enumerate(self._feetLinkId):
                    if contact[4] == linkId:
                        feet_contact_states[k] = 1

        feetVelocity = []
        feetForce = []

        for footId in self._feetLinkId:
            feetVelocity.append(self._p.getLinkState(self._quadruped, footId, 1)[6])
            feetForce.append(self._p.getJointState(self._quadruped, footId)[2][:3])

        self._feetForce = feetForce
        self._feetVelocity = feetVelocity
        self._basePos = pos
        self._baseOrn = orn
        self._jointsPos = joints_position
        self._jointsVelocity = joints_velocity
        self._jointsTorque = joints_torque
        self._contact_feet = feet_contact_states
        self._baseVelocity = pos_v
        self._baseAngularVelocity = orn_v
        self._observation = self.getExtendedObservation()

    def step(self, action: np.ndarray, use_residual=False):
        """
        Here action at is the desired join position for the 12 robot joints.
        """
        self._actions = action * self.params.action_bound
        for j, a in enumerate(self._actions):
            self._p.setJointMotorControl2(self._quadruped, self._jointIds[j],
                                          self._p.POSITION_CONTROL, a, force=self._maxForce)

        self._p.stepSimulation()
        self._update_states()
        self._envStepCounter += 1
        self._observation = self.getExtendedObservation()
        reward = self._reward()
        done = self._termination()

        return np.array(self._observation), reward, done, {}

    def render(self, mode='human', close=False):

        if mode != "rgb_array":
            return np.array([])

        base_pos, _ = self._p.getBasePositionAndOrientation(self._quadruped)
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
        One episode would terminate when the height of the body will be below 0.28m (z < 0.28)
        or the rotation in the body
        """
        termination = False

        if self._basePos[2] < 0.28 or self._baseOrn[0] > 0.4 or self._baseOrn[1] > 0.2:
            termination = True

        return termination
