import math
import gym
from lib.utils import states2observations
import numpy as np


class LinearPhysicsParams:
    def __init__(self):
        self.ini_states = [0.0, 0.0, 0.1, 0.0, False]
        self.x_threshold = 0.3
        self.length = 0.64
        self.theta_dot_threshold = 15


class LinearPhysics(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30  # this is not working
    }

    def __init__(self, params: LinearPhysicsParams):

        self.states = None
        self.params = params
        self.matrix_A = np.array([[1, 0.03333333, 0, 0],
                                   [0.13898557, 1.15349157, 1.4669255, 0.20617051],
                                   [0, 0, 1, 0.03333333],
                                   [-0.30894424, -0.34118891, -2.53462563, 0.54171366]])
        self.viewer = None

    def step(self, action=None):
        """
        Dynamics
        """
        x, x_dot, theta, theta_dot, _ = self.states

        current_states = np.transpose([x, x_dot, theta, theta_dot])  # 4 x 1
        next_states = self.matrix_A @ current_states
        x, x_dot, theta, theta_dot = np.squeeze(next_states).tolist()
        theta_rescale = math.atan2(math.sin(theta), math.cos(theta))  # to rescale theta into [-pi, pi)

        failed = self.is_failed(x, theta_dot)

        new_states = [x, x_dot, theta_rescale, theta_dot, failed]
        self.states = new_states  # to update animation
        return self.states

    def reset(self, reset_states=None):
        if reset_states is not None:
            self.states = reset_states
        else:
            self.states = self.params.ini_states

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

    def get_shape_observations(self):
        observations, _ = states2observations(self.params.ini_states)
        return len(observations)
