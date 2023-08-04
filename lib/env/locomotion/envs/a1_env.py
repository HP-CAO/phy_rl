import scipy.interpolate
import numpy as np
import pybullet_data
from pybullet_utils import bullet_client
import pybullet  # pytype:disable=import-error
import math

from lib.env.locomotion.agents.whole_body_controller import com_velocity_estimator
from lib.env.locomotion.agents.whole_body_controller import gait_generator as gait_generator_lib
from lib.env.locomotion.agents.whole_body_controller import locomotion_controller
from lib.env.locomotion.agents.whole_body_controller import openloop_gait_generator
from lib.env.locomotion.agents.whole_body_controller import raibert_swing_leg_controller
from lib.env.locomotion.agents.whole_body_controller import torque_stance_leg_controller

from lib.env.locomotion.robots import a1
from lib.env.locomotion.robots import robot_config
from dm_control.utils import rewards

_NUM_SIMULATION_ITERATION_STEPS = 300
_MAX_TIME_SECONDS = 30.
_STANCE_DURATION_SECONDS = [0.3] * 4  # For faster trotting (v > 1.5 ms reduce this to 0.13s).

_DUTY_FACTOR = [0.6] * 4

_INIT_PHASE_FULL_CYCLE = [0.9, 0, 0, 0.9]
_INIT_LEG_STATE = (
    gait_generator_lib.LegState.SWING,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.SWING,
)


def _setup_controller(robot):
    """Demonstrates how to create a locomotion controller."""
    desired_speed = (0, 0)
    desired_twisting_speed = 0

    gait_generator = openloop_gait_generator.OpenloopGaitGenerator(
        robot,
        stance_duration=_STANCE_DURATION_SECONDS,
        duty_factor=_DUTY_FACTOR,
        initial_leg_phase=_INIT_PHASE_FULL_CYCLE,
        initial_leg_state=_INIT_LEG_STATE)

    window_size = 20  # the window_size is 20 for simulation training if not FLAGS.use_real_robot else 60
    state_estimator = com_velocity_estimator.COMVelocityEstimator(robot, window_size=window_size)

    sw_controller = raibert_swing_leg_controller.RaibertSwingLegController(
        robot,
        gait_generator,
        state_estimator,
        desired_speed=desired_speed,
        desired_twisting_speed=desired_twisting_speed,
        desired_height=robot.MPC_BODY_HEIGHT,
        foot_clearance=0.01)

    st_controller = torque_stance_leg_controller.TorqueStanceLegController(
        robot,
        gait_generator,
        state_estimator,
        desired_speed=desired_speed,
        desired_twisting_speed=desired_twisting_speed,
        desired_body_height=robot.MPC_BODY_HEIGHT)

    controller = locomotion_controller.LocomotionController(
        robot=robot,
        gait_generator=gait_generator,
        state_estimator=state_estimator,
        swing_leg_controller=sw_controller,
        stance_leg_controller=st_controller,
        clock=robot.GetTimeSinceReset)
    return controller


def _update_controller_params(controller, lin_speed, ang_speed):
    controller.swing_leg_controller.desired_speed = lin_speed
    controller.swing_leg_controller.desired_twisting_speed = ang_speed
    controller.stance_leg_controller.desired_speed = lin_speed
    controller.stance_leg_controller.desired_twisting_speed = ang_speed


def _generate_example_linear_angular_speed(t):
    """Creates an example speed profile based on time for demo purpose."""
    vx = 0.6
    vy = 0.2
    wz = 0.8

    time_points = (0, 5, 10, 15, 20, 25, 30)
    speed_points = ((0, 0, 0, 0), (0, 0, 0, wz), (vx, 0, 0, 0), (0, 0, 0, -wz),
                    (0, -vy, 0, 0), (0, 0, 0, 0), (0, 0, 0, wz))

    speed = scipy.interpolate.interp1d(time_points,
                                       speed_points,
                                       kind="previous",
                                       fill_value="extrapolate",
                                       axis=0)(t)
    return speed[0:3], speed[3], False


class A1Params:
    def __init__(self):
        self.show_gui = True
        self.time_step = 0.002  # 1 / control frequency
        self.if_add_terrain = False
        self.random_reset_eval = False
        self.random_reset_train = False
        self.if_record_video = False
        self.action_magnitude = 1
        self.reward_type = 'velocity'


class A1Robot:
    def __init__(self, params: A1Params):
        self.params = params

        if self.params.show_gui:
            self.p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self.p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)

        self.robot = None
        self.mpc_control = None

        self.states_observations_dim = 12
        self.action_dim = 6

        self.termination = None
        self.states = None
        self.observation = None
        self.target_lin_speed = [1.0, 0, 0]
        self.target_ang_speed = 0.0
        self.diff_q = None
        self.diff_dq = None
        self.current_step = 0
        self.previous_tracking_error = None
        self.initialize_env()
        self.states_vector = []
        self.height_vector = []

    def random_reset(self):
        pass

    def reset(self, step, reset_status=None):
        self.p.resetSimulation()
        self.p.setPhysicsEngineParameter(numSolverIterations=30)
        # self.p.setTimeStep(self.params.time_step)
        self.p.setTimeStep(0.001)
        self.p.setGravity(0, 0, -9.8)
        self.p.setPhysicsEngineParameter(enableConeFriction=0)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane = self.p.loadURDF("./lib/env/locomotion/envs/meshes/plane.urdf")

        # self.p.changeDynamics(plane, -1, lateralFriction=0.575)  # change friction from higher to lower
        self.p.changeDynamics(plane, -1, lateralFriction=0.44)  # change friction from higher to lower

        if self.params.if_record_video:
            self.p.startStateLogging(self.p.STATE_LOGGING_VIDEO_MP4, f"{step}_record.mp4")

        self.robot = a1.A1(self.p,
                           motor_control_mode=robot_config.MotorControlMode.HYBRID,
                           enable_action_interpolation=False,
                           reset_time=2,
                           time_step=0.002,
                           action_repeat=1)

        if self.params.if_add_terrain:
            self.add_terrain()

        self.mpc_control = _setup_controller(self.robot)  # MPC controller for low-level control
        self.mpc_control.reset()
        self.states = self.get_state()
        self.previous_tracking_error = self.get_tracking_error()
        self.observation, self.termination, _ = self.get_observations(self.states)
        self.current_step = 0

    def get_observations(self, state):
        observation = []  # 16 dims
        roll, pitch, _ = state['base_rpy']

        termination = False
        abort = False

        angle_threshold = 30 * (math.pi / 180)

        # observation of root orientation,  3
        robot_orientation = state['base_rpy']
        observation.extend(robot_orientation)

        # root angular velocity 3
        robot_angular_velocity = state['base_rpy_rate']
        observation.extend(robot_angular_velocity)

        # linear_velocity 3
        robot_linear_velocity = state['base_vel']
        observation.extend(robot_linear_velocity)

        # # motion_angle 12
        # motor_angle = state['motor_angles']
        # observation.extend(motor_angle)
        #
        # # motion angle rate 12
        # motor_angle_rate = state['motor_vels']
        # observation.extend(motor_angle_rate)

        # foot_contact = 4
        foot_contact = state['contacts']
        observation.extend(foot_contact)

        # velocity in body frame 3
        velocity_in_body_frame = state["base_vels_body_frame"]
        observation.extend(velocity_in_body_frame)

        # if abs(roll) > angle_threshold or abs(pitch) > angle_threshold:
        #     print("roll", roll, "pitch", pitch)
        #     termination = True

        confall = self.mpc_control.stance_leg_controller.estimate_robot_x_y_z()

        fall_threshold = 0.12

        # if abs(roll) > angle_threshold or abs(pitch) > angle_threshold:

        if abs(confall[2]) < fall_threshold:
            print("Fall: height:", confall[2])
            termination = True

        if math.isnan(float(robot_linear_velocity[0])):
            abort = True
            print("ABORT_DUE_TO_SIMULATION_ERROR")
        return observation, termination, abort

    def get_state(self):

        states = dict(timestamp=self.robot.GetTimeSinceReset(),
                      base_rpy=self.robot.GetBaseRollPitchYaw(),
                      motor_angles=self.robot.GetMotorAngles(),
                      base_vel=self.robot.GetBaseVelocity(),
                      base_vels_body_frame=self.mpc_control.state_estimator.com_velocity_body_frame,
                      base_rpy_rate=self.robot.GetBaseRollPitchYawRate(),
                      motor_vels=self.robot.GetMotorVelocities(),
                      contacts=self.robot.GetFootContacts())

        return states

    def get_states_vector(self):
        angle = self.states['base_rpy']
        com_position_xyz = self.mpc_control.stance_leg_controller.estimate_robot_x_y_z()
        base_rpy_rate = self.states['base_rpy_rate']
        com_velocity = self.states['base_vels_body_frame']
        states_vector = np.hstack((com_position_xyz, angle, com_velocity, base_rpy_rate))
        # states_vector = np.hstack((angle, com_position_xyz, base_rpy_rate, com_velocity))
        return states_vector

    def get_tracking_error(self):  # this is used for computing reward
        reference_vx = 1.0
        reference_p_z = 0.24
        # reference_vector = np.array([0., 0., 0., 0, 0, reference_p_z, 0., 0., 0., reference_vx, 0., 0.])

        reference_vector = np.array([0, 0, reference_p_z, 0., 0., 0., reference_vx, 0., 0., 0., 0., 0.])

        states_vector_robot = self.get_states_vector()
        tracking_error = states_vector_robot - reference_vector
        return tracking_error

    def get_run_reward(self, x_velocity: float, move_speed: float, cos_pitch: float, dyaw: float):
        reward = rewards.tolerance(cos_pitch * x_velocity,
                                   bounds=(move_speed, 2 * move_speed),
                                   margin=2 * move_speed,
                                   value_at_margin=0,
                                   sigmoid='linear')
        reward -= 0.1 * np.abs(dyaw)
        return 10 * reward  # [0, 1] => [0, 10]

    def get_drl_reward(self):  # todo change the reward to be consistent as MPC, get rid of the first 2 terms
        x_velocity = self.states['base_vels_body_frame'][0]
        move_speed = self.target_lin_speed[0]
        cos_pitch = math.cos(self.states['base_rpy'][1])
        dyaw = self.states['base_rpy'][2]

        reward = self.get_run_reward(x_velocity, move_speed, cos_pitch, dyaw)
        return reward

    def get_reward(self):
        if self.params.reward_type == 'velocity':
            return self.get_drl_reward()
        else:
            return self.get_ly_reward()

    def get_ly_reward(self):

        # p_vector = [0, 0, 0, 0, 0, 0, 1.81666331e-04, 1.81892955e-04,
        #             1.87235756e-04, 1, 1.88585412e-04, 1.88390268e-04]
        #
        # p_matrix = np.diag(p_vector)

        p_matrix = np.array([[6.3394, 0, 0, 0, 0, 0, 0.4188, 0, 0, 0, 0, 0],
                             [0, 1.4053, 0, 0, 0, 0, 0, 0.3018, 0, 0, 0, 0],
                             [0, 0, 94.0914, 0, 0, 0, 0, 0, 9.1062, 0, 0, 0],
                             [0, 0, 0, 95.1081, 0, 0, 0, 0, 0, 9.2016, 0, 0],
                             [0, 0, 0, 0, 95.1081, 0, 0, 0, 0, 0, 9.2016, 0],
                             [0, 0, 0, 0, 0, 1.4053, 0, 0, 0, 0, 0, 0.3018],
                             [0.4188, 0, 0, 0, 0, 0, 106.1137, 0, 0, 0, 0, 0],
                             [0, 0.3018, 0, 0, 0, 0, 0, 77.1735, 0, 0, 0, 0],
                             [0, 0, 9.1062, 0, 0, 0, 0, 0, 1.8594, 0, 0, 0],
                             [0, 0, 0, 9.2016, 0, 0, 0, 0, 0, 1.8783, 0, 0],
                             [0, 0, 0, 0, 9.2016, 0, 0, 0, 0, 0, 1.8783, 0],
                             [0, 0, 0, 0, 0, 0.3018, 0, 0, 0, 0, 0, 77.1735]]) * 1

        M_matrix = np.array([[6.33931716274651, 0, 0, 0, 0, 0, 0.39824214223179, 0, 0, 0, 0, 0],
                             [0, 1.40521824475728, 0, 0, 0, 0, 0, 0.286679284833682, 0, 0, 0, 0],
                             [0, 0, 92.2887010538464, 0, 0, 0, 0, 0, 8.92428326269013, 0, 0, 0],
                             [0, 0, 0, 93.2865880895433, 0, 0, 0, 0, 0, 9.01777538552449, 0, 0],
                             [0, 0, 0, 0, 93.2865880895433, 0, 0, 0, 0, 0, 9.01777538552449, 0],
                             [0, 0, 0, 0, 0, 1.40521824475728, 0, 0, 0, 0, 0, 0.286679284833682],
                             [0.39824214223179, 0, 0, 0, 0, 0, 97.7952232108596, 0, 0, 0, 0, 0],
                             [0, 0.286679284833682, 0, 0, 0, 0, 0, 72.6131010296885, 0, 0, 0, 0],
                             [0, 0, 8.92428326269013, 0, 0, 0, 0, 0, 1.84054305542176, 0, 0, 0],
                             [0, 0, 0, 9.01777538552449, 0, 0, 0, 0, 0, 1.8592311555769, 0, 0],
                             [0, 0, 0, 0, 9.01777538552449, 0, 0, 0, 0, 0, 1.8592311555769, 0],
                             [0, 0, 0, 0, 0, 0.286679284833682, 0, 0, 0, 0, 0, 72.6131010296885]]) * 1

        tracking_error_current = self.get_tracking_error()
        tracking_error_current = np.expand_dims(tracking_error_current, axis=-1)
        tracking_error_pre = self.previous_tracking_error
        tracking_error_pre = np.expand_dims(tracking_error_pre, axis=-1)

        ly_reward_cur = np.transpose(tracking_error_current, axes=(1, 0)) @ p_matrix @ tracking_error_current
        ly_reward_pre = np.transpose(tracking_error_pre, axes=(1, 0)) @ M_matrix @ tracking_error_pre

        # ly_reward_pre = np.transpose(tracking_error_pre, axes=(1, 0)) @ p_matrix @ tracking_error_pre

        reward = ((ly_reward_pre - ly_reward_cur) * 0.01)

        return reward

    def initialize_env(self):
        self.reset(step=0)

    def step(self, action, action_mode='mpc'):
        """
        Here the action is generated from DRL agent, that controls ground reaction force (GRF).
        dim: 12, 3 dims (motors) for each leg action is in [-1,1]
        """
        self.previous_tracking_error = self.get_tracking_error()

        if action_mode == 'residual':
            _update_controller_params(self.mpc_control, self.target_lin_speed, self.target_ang_speed)
            self.mpc_control.update()  # update the clock
            # rescale the action to be [0.5, 1.5], this will be a multiplier to scale up/down the mpc action
            action *= self.params.action_magnitude
            applied_action, _, diff_q, diff_dq = self.mpc_control.get_action(action, mode="residual")
            self.diff_q = diff_q
            self.diff_dq = diff_dq
        elif action_mode == 'mpc':
            _update_controller_params(self.mpc_control, self.target_lin_speed, self.target_ang_speed)
            self.mpc_control.update()  # update the clock
            # applied_action, _, diff_q, diff_dq = self.mpc_control.get_action()
            applied_action, _, diff_q, diff_dq = self.mpc_control.get_action(mode="mpc")
        elif action_mode == 'drl':
            action *= self.params.action_magnitude  # to check the dim and magnitude of the action
            _update_controller_params(self.mpc_control, self.target_lin_speed, self.target_ang_speed)
            self.mpc_control.update()
            applied_action, _, diff_q, diff_dq = self.mpc_control.get_action(action, mode="drl")
        else:
            raise NameError
        self.robot.Step(applied_action)

        state = self.get_state()
        observation, termination, abort = self.get_observations(state)
        self.states = state  # update the states buffer
        self.observation = observation
        self.termination = termination
        self.states_vector.append(self.states["base_vels_body_frame"][0])
        self.height_vector.append(self.mpc_control.stance_leg_controller.estimate_robot_x_y_z()[-1])

        self.current_step += 1
        return observation, termination, abort

    def get_performance_score(self):
        pass

    def add_terrain(self):
        boxHalfLength = 0.2
        boxHalfWidth = 2.5
        boxHalfHeight = 0.05
        sh_colBox = self.p.createCollisionShape(self.p.GEOM_BOX, halfExtents=[0.5, boxHalfWidth, 0.05])
        sh_final_col = self.p.createCollisionShape(self.p.GEOM_BOX, halfExtents=[0.5, boxHalfWidth, 0.05])
        boxOrigin = 0.8 + boxHalfLength
        step1 = self.p.createMultiBody(baseMass=0, baseCollisionShapeIndex=sh_colBox,
                                       basePosition=[boxOrigin, 1, boxHalfHeight],
                                       baseOrientation=[0.0, 0.0, 0.0, 1])

        step2 = self.p.createMultiBody(baseMass=0, baseCollisionShapeIndex=sh_final_col,
                                       basePosition=[boxOrigin + 0.5 + boxHalfLength, 1, 0.05 + 2 * boxHalfHeight],
                                       baseOrientation=[0.0, 0.0, 0.0, 1])

        self.p.changeDynamics(step1, -1, lateralFriction=0.85)
        self.p.changeDynamics(step2, -1, lateralFriction=0.85)
