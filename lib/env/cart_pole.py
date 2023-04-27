import math
import gym
from gym.utils import seeding
import numpy as np
import copy


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
        self.ini_states = [0.1, 0.1, 0.15, 0.0, False]
        self.targets = [0., 0.]

        self.distance_score_factor = 0
        self.tracking_error_factor = 1
        self.lyapunov_reward_factor = 1
        self.high_performance_reward_factor = 0.5
        self.action_penalty = 0
        self.crash_penalty = 0

        self.observe_reference_states = False
        self.random_reset_train = True
        self.random_reset_eval = False
        self.update_reference_model = True
        self.sparse_reset = False
        self.use_ubc_lya_reward = True
        self.add_uu_dis = False


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
        self.states_observations_refer_dim = 4  # error between the states of ref and real
        self.action_dim = 1  # force input or voltage

        self.matrix_A = np.array([[1, 0.03333333, 0, 0],
                                  [0.0247, 1.1204, 1.1249, 0.2339],
                                  [0, 0, 1, 0.03333333],
                                  [-0.0580, -0.2822, -1.8709, 0.4519]])
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

    def step(self, action: float, use_residual=False):
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

        if use_residual:
            F = np.array([8.25691599, 6.76016534, 40.12484514, 6.84742553])
            # F = [24.3889996, 38.80261887, 195.86059044, 33.26717081]
            force_res = F[0] * x + F[1] * x_dot + F[2] * theta + F[3] * theta_dot  # residual control commands

            # force_res = 0.7400 * x + 3.6033 * x_dot + 35.3534 * theta + 6.9982 * theta_dot  # residual control commands
            force = force + force_res  # RL control commands + residual control commands

        # force = np.clip(force, a_min=-5 * self.params.force_mag, a_max=5 * self.params.force_mag)

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

        uu1 = 0
        uu2 = 0

        if self.params.add_uu_dis:
            uu1, uu2 = get_unk_unk_dis()

        if self.params.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * (xacc + uu1)  # here we inject disturbances
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * (thetaacc + uu2)  # here we inject disturbances
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
        if self.params.sparse_reset:
            ran_x = 0.10
            ran_v = 0.10
            ran_theta = 0.15
            ran_theta_v = 0
            failed = False
            sam_inx = np.random.randint(9, size=1)

            if sam_inx == 0:
                self.states = [ran_x, ran_v, ran_theta, ran_theta_v, failed]
            elif sam_inx == 1:
                self.states = [-ran_x, ran_v, ran_theta, ran_theta_v, failed]
            elif sam_inx == 2:
                self.states = [ran_x, -ran_v, ran_theta, ran_theta_v, failed]
            elif sam_inx == 3:
                self.states = [ran_x, ran_v, -ran_theta, ran_theta_v, failed]
            elif sam_inx == 4:
                self.states = [-ran_x, -ran_v, ran_theta, ran_theta_v, failed]
            elif sam_inx == 5:
                self.states = [-ran_x, ran_v, -ran_theta, ran_theta_v, failed]
            elif sam_inx == 6:
                self.states = [ran_x, -ran_v, -ran_theta, ran_theta_v, failed]
            elif sam_inx == 7:
                self.states = [-ran_x, -ran_v, -ran_theta, ran_theta_v, failed]
            elif sam_inx == 8:
                self.states = [-ran_x, -ran_v, 0.10, ran_theta_v, failed]
        else:
            ran_x = np.random.uniform(-0.85, 0.85)
            ran_v = np.random.uniform(-0.4, 0.4)
            ran_theta = np.random.uniform(-0.6, 0.6)
            ran_theta_v = np.random.uniform(-0.4, 0.4)
            failed = False
            self.states = [ran_x, ran_v, ran_theta, ran_theta_v, failed]
        self.states_refer = copy.deepcopy(self.states)

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
        :param voltage: voltage from the agent
        :return: force actuation to the plant
        """
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
    def get_lyapunov_reward(P_matrix, states_real):
        state = np.array(states_real[0:4])
        state = np.expand_dims(state, axis=0)
        Lya1 = np.matmul(state, P_matrix)
        Lya = np.matmul(Lya1, np.transpose(state))
        return Lya

    @staticmethod
    def get_tracking_error(P_matrix, states_real, states_reference):

        state = np.array(states_real[0:4])
        state = np.expand_dims(state, axis=0)
        state_ref = np.array(states_reference[0:4])
        state_ref = np.expand_dims(state_ref, axis=0)

        state_error = state - state_ref
        eLya1 = np.matmul(state_error, P_matrix)
        eLya = np.matmul(eLya1, np.transpose(state_error))

        error = -eLya

        return error

    def reward_fcn(self, states_current, action, states_next, states_refer_current):

        P_matrix = np.array([[4.6074554, 1.49740096, 5.80266046, 0.99189224],
                             [1.49740096, 0.81703147, 2.61779592, 0.51179642],
                             [5.80266046, 2.61779592, 11.29182733, 1.87117709],
                             [0.99189224, 0.51179642, 1.87117709, 0.37041435]])  # new Lyapunov P matrix

        S_matrix = np.array([[1, 0.03333333, 0, 0, ],
                             [0.27592037, 1.22590363, 1.2843559, 0.2288196],
                             [0, 0, 1, 0.03333333],
                             [-0.64668827, -0.52946156, -2.24458365, 0.46370415]])  # new system matrix

        observations, _ = states2observations(states_current)
        targets = self.params.targets  # [0, 0] stands for position and angle

        terminal = states_next[-1]

        distance_score = self.get_distance_score(observations, targets)
        distance_reward = distance_score * self.params.high_performance_reward_factor

        lyapunov_reward_current = self.get_lyapunov_reward(P_matrix, states_current)

        ##########
        tem_state_a = np.array(states_current[0:4])
        tem_state_b = np.expand_dims(tem_state_a, axis=0)
        tem_state_c = np.matmul(tem_state_b, np.transpose(S_matrix))
        tem_state_d = np.matmul(tem_state_c, P_matrix)
        lyapunov_reward_current_aux = np.matmul(tem_state_d, np.transpose(tem_state_c))
        ###########

        lyapunov_reward_next = self.get_lyapunov_reward(P_matrix, states_next)

        if self.params.use_ubc_lya_reward:
            lyapunov_reward = lyapunov_reward_current - lyapunov_reward_next
        else:
            lyapunov_reward = lyapunov_reward_current_aux - lyapunov_reward_next  # ours

        lyapunov_reward *= self.params.lyapunov_reward_factor

        action_penalty = -1 * self.params.action_penalty * action * action

        r = distance_reward + lyapunov_reward + action_penalty

        return r, distance_score


def states2observations(states):
    x, x_dot, theta, theta_dot, failed = states
    observations = [x, x_dot, math.sin(theta), math.cos(theta), theta_dot]
    return observations, failed


def observations2states(observations, failed):
    x, x_dot, s_theta, c_theta, theta_dot = observations[:5]
    states = [x, x_dot, np.arctan2(s_theta, c_theta), theta_dot, failed]
    return states


def get_unk_unk_dis():
    rng = np.random.default_rng(seed=1)
    a = 11 * np.random.random(1)  # [0, 11]
    b = 11 * np.random.random(1)  # [0, 11]
    uu1 = -rng.beta(a, b) + rng.beta(a, b)
    uu2 = -rng.beta(a, b) + rng.beta(a, b)
    uu1 *= 2  # [-2, 2]
    uu2 *= 2  # [-2, 2]
    return uu1[0], uu2[0]
