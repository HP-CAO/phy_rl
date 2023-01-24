import math
import gym
from gym.utils import seeding
import numpy as np


class CartpoleParams:
    def __init__(self):
        self.x_threshold = 0.3
        self.theta_dot_threshold = 15
        self.kinematics_integrator = 'euler'
        self.gravity = 9.8
        self.mass_cart = 0.94
        self.mass_pole = 0.23
        self.force_mag = 5.0
        self.voltage_mag = 5.0
        self.length = 0.64
        self.theta_random_std = 0.8
        self.friction_cart = 10
        self.friction_pole = 0.0011
        self.simulation_frequency = 30

        self.with_friction = True
        self.force_input = True
        self.ini_states = [0.0, 0.0, 0.1, 0.0, False]
        self.targets = [0., 0.]

        self.distance_score_factor = 1
        self.tracking_error_factor = 1
        self.lyapunov_reward_factor = 1
        self.action_penalty = 0.05
        self.crash_penalty = 10

        self.observe_reference_states = False
        self.random_reset_train = True
        self.random_reset_eval = False
        self.update_reference_model = True


class Cartpole(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, params: CartpoleParams):

        self.params = params
        self.total_mass = (self.params.mass_cart + self.params.mass_pole)
        self.half_length = self.params.length * 0.5
        self.pole_mass_length_half = self.params.mass_pole * self.half_length
        self.tau = 1 / self.params.simulation_frequency

        self.seed()
        self.viewer = None
        self.states = None
        self.steps_beyond_terminal = None

        self.states_dim = 4  # x, x_dot, theta, theta_dot
        self.states_observations_dim = 5  # x, x_dot, s_theta, c_theta, theta_dot
        self.states_observations_refer_dim = 5
        self.action_dim = 1  # force input or voltage

        self.matrix_A = np.array([[1, 0.03333333, 0, 0],
                                  [0.13898557, 1.15349157, 1.4669255, 0.20617051],
                                  [0, 0, 1, 0.03333333],
                                  [-0.30894424, -0.34118891, -2.53462563, 0.54171366]])
        self.states_refer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def refer_step(self):
        x, x_dot, theta, theta_dot, _ = self.states_refer
        current_states = np.transpose([x, x_dot, theta, theta_dot])  # 4 x 1
        next_states = self.matrix_A @ current_states
        x, x_dot, theta, theta_dot = np.squeeze(next_states).tolist()
        theta_rescale = math.atan2(math.sin(theta), math.cos(theta))  # to rescale theta into [-pi, pi)
        failed = self.is_failed(x, theta_dot)
        new_states = [x, x_dot, theta_rescale, theta_dot, failed]
        self.states_refer = new_states  # to update animation
        return self.states_refer

    def step(self, action: float):
        """
        param: action: a scalar value (not numpy type) [-1,1]
        return: a list of states
        """

        x, x_dot, theta, theta_dot, _ = self.states

        if self.params.force_input:
            force = action * self.params.force_mag
        else:
            voltage = action * self.params.voltage_mag
            force = self.voltage2force(voltage, x_dot)

        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # kinematics of the inverted pendulum
        if self.params.with_friction:
            """ with friction"""
            temp \
                = (force + self.pole_mass_length_half * theta_dot ** 2 *
                   sintheta - self.params.friction_cart * x_dot) / self.total_mass

            thetaacc = \
                (self.params.gravity * sintheta - costheta * temp -
                 self.params.friction_pole * theta_dot / self.pole_mass_length_half) / \
                (self.half_length * (4.0 / 3.0 - self.params.mass_pole * costheta ** 2 / self.total_mass))

            xacc = temp - self.pole_mass_length_half * thetaacc * costheta / self.total_mass

        else:
            """without friction"""
            temp = (force + self.pole_mass_length_half * theta_dot ** 2 * sintheta) / self.total_mass
            thetaacc = (self.params.gravity * sintheta - costheta * temp) / \
                       (self.half_length * (4.0 / 3.0 - self.params.mass_pole * costheta ** 2 / self.total_mass))
            xacc = temp - self.pole_mass_length_half * thetaacc * costheta / self.total_mass

        if self.params.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
            failed = self.is_failed(x, theta_dot)

        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
            failed = self.is_failed(x, theta_dot)

        theta_rescale = math.atan2(math.sin(theta), math.cos(theta))
        new_states = [x, x_dot, theta_rescale, theta_dot, failed]
        self.states = new_states  # to update animation
        return self.states

    def reset(self, reset_states=None):
        if reset_states is not None:
            self.states = reset_states
            self.states_refer = reset_states
        else:
            self.states = self.params.ini_states
            self.states_refer = self.states

    def random_reset(self):
        # todo here we only randomize theta
        # ran_x = np.random.uniform(-0.8 * self.params.x_threshold, 0.8 * self.params.x_threshold)
        # if self.is_failed(ran_x, 0):
        #    ran_x += 0
        ran_x = 0.
        ran_v = 0.
        ran_theta = np.random.uniform(-0.1, 0.1)
        # ran_theta = np.random.normal(math.pi, self.params.theta_random_std)
        ran_theta_v = 0.
        failed = False
        self.states = [ran_x, ran_v, ran_theta, ran_theta_v, failed]
        self.states_refer = self.states

    def render(self, mode='human', states=None, is_normal_operation=True):

        screen_width = 600
        screen_height = 400
        world_width = self.params.x_threshold * 2 + 1
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * self.params.length
        cartwidth = 50.0
        cartheight = 30.0
        target_width = 45
        target_height = 45

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            # target
            self.targettrans = rendering.Transform()
            target = rendering.Image('./target.svg', width=target_width, height=target_height)
            target.add_attr(self.targettrans)
            self.viewer.add_geom(target)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            if not is_normal_operation:
                cart.set_color(1.0, 0, 0)
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)
            self._pole_geom = pole

        if states is None:
            if self.states is None:
                return None
            else:
                x = self.states
        else:
            x = states

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        targetx = 0 * scale + screen_width / 2.0
        targety = polelen + carty

        self.carttrans.set_translation(cartx, carty)
        self.targettrans.set_translation(targetx, targety)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def is_failed(self, x, theta_dot):
        failed = bool(x <= -self.params.x_threshold
                      or x >= self.params.x_threshold
                      or theta_dot > self.params.theta_dot_threshold)
        return failed

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def voltage2force(self, voltage, cart_v):
        """
        The equation can be found in the Pendulum Gantry workbook eq 2.11
        Convert voltage control to force control
        :param voltage: voltage action from the agent
        :return: force actuation to the plant
        """

        # f = 0.90 * 3.71 * 7.68 * (voltage * 6.35 * 0.69  - 7.68 * 3.71 * cart_v) / (6.35 * 6.35 * 2.6)
        f = 1.07 * voltage - 6.96 * cart_v
        return f

    def get_distance_score(self, observations, target):
        distance_score_factor = 5  # to adjust the exponential gradients
        cart_position = observations[0]
        pendulum_angle_sin = observations[2]
        pendulum_angle_cos = observations[3]

        target_cart_position = target[0]
        target_pendulum_angle = target[1]

        pendulum_length = self.params.length

        pendulum_tip_position = np.array(
            [cart_position + pendulum_length * pendulum_angle_sin, pendulum_length * pendulum_angle_cos])

        target_tip_position = np.array(
            [target_cart_position + pendulum_length * np.sin(target_pendulum_angle),
             pendulum_length * np.cos(target_pendulum_angle)])

        distance = np.linalg.norm(target_tip_position - pendulum_tip_position)

        distance_score = np.exp(-distance * distance_score_factor)
        return distance_score

    @staticmethod
    def get_lyapunov_reward(states_real):
        # here the states are compared to [0, 0, 0, 0]
        rx_squared_error = (states_real[0]) ** 2
        rv_squared_error = (states_real[1]) ** 2
        rtheta_squared_error = (states_real[2]) ** 2
        rtheta_dot_squared_error = (states_real[3]) ** 2

        distance \
            = np.exp(-1 * (rx_squared_error + rv_squared_error + rtheta_squared_error + rtheta_dot_squared_error) * 2)

        return distance

    @staticmethod
    def get_tracking_error(states_real, states_reference):
        x_squared_error = (states_real[0] - states_reference[0]) ** 2
        v_squared_error = (states_real[1] - states_reference[1]) ** 2
        theta_squared_error = (states_real[2] - states_reference[2]) ** 2
        theta_dot_squared_error = (states_real[3] - states_reference[3]) ** 2
        error = -1 * (x_squared_error + v_squared_error + theta_squared_error + theta_dot_squared_error)

        return error

    def reward_fcn(self, states_current, action, states_next, states_refer_current):
        observations, _ = states2observations(states_current)
        targets = self.params.targets  # [0, 0] stands for position and angle

        terminal = states_next[-1]

        distance_score = self.get_distance_score(observations, targets) * self.params.distance_score_factor
        lyapunov_reward = self.get_lyapunov_reward(states_current) * self.params.lyapunov_reward_factor
        tracking_error = self.get_tracking_error(states_current, states_refer_current) * self.params.tracking_error_factor
        action_penalty = -1 * self.params.action_penalty * action
        crash_penalty = -1 * self.params.crash_penalty * terminal

        r = distance_score + lyapunov_reward + tracking_error + action_penalty + crash_penalty
        return r, distance_score


def states2observations(states):
    x, x_dot, theta, theta_dot, failed = states
    observations = [x, x_dot, math.sin(theta), math.cos(theta), theta_dot]
    return observations, failed


def observations2states(observations, failed):
    x, x_dot, s_theta, c_theta, theta_dot = observations[:5]
    states = [x, x_dot, np.arctan2(s_theta, c_theta), theta_dot, failed]
    return states