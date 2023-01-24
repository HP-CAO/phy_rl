import numpy as np
import math


class RewardParams:
    def __init__(self):
        self.distance_score_reward = 0
        self.action_penalty = 0
        self.crash_penalty = 0
        self.distance_score_factor = 0

        self.x_error_weight = 1
        self.v_error_weight = 1
        self.theta_error_weight = 1
        self.theta_dot_error_weight = 1
        self.tracking_error_weight = 1


class RewardFcn:

    def __init__(self, params: RewardParams):
        self.params = params
        self.reward = self.distance_reward

    def distance_reward(self, observations, targets, action, terminal, pole_length, states_real):
        """
        calculate reward
        :param pole_length: the length of the pole
        :param observations: [pos, vel, sin_angle, cos_angle, angle_rate]
        :param targets: [pos_target, angle_target]
        :param action: action based on current states
        :param terminal: crash or not
        :return: a scalar value
        """

        distance_score = self.get_distance_score(observations, targets, pole_length, self.params.distance_score_factor,
                                                 states_real)

        r = self.params.distance_score_reward * distance_score
        r -= self.params.action_penalty * action
        r -= self.params.crash_penalty * terminal

        return r

    @staticmethod
    def get_distance_score(observation, target, pole_length, distance_score_factor, states_real):
        """
        calculate reward
        :param pole_length: the length of the pole
        :param distance_score_factor: co-efficients of the distance score
        :param observation: [pos, vel, sin_angle, cos_angle, angle_rate]
        :param target: [pos_target, angle_target]
        """

        cart_position = observation[0]
        pendulum_angle_sin = observation[2]
        pendulum_angle_cos = observation[3]

        target_cart_position = target[0]
        target_pendulum_angle = target[1]

        pendulum_length = pole_length

        pendulum_tip_position = np.array(
            [cart_position + pendulum_length * pendulum_angle_sin, pendulum_length * pendulum_angle_cos])

        target_tip_position = np.array(
            [target_cart_position + pendulum_length * np.sin(target_pendulum_angle),
             pendulum_length * np.cos(target_pendulum_angle)])

        # distance = np.linalg.norm(target_tip_position - pendulum_tip_position)

        rx_squared_error = (states_real[0]) ** 2
        rv_squared_error = (states_real[1]) ** 2
        rtheta_squared_error = (states_real[2]) ** 2
        rtheta_dot_squared_error = (states_real[3]) ** 2

        distance = 0 * np.exp(
            -1 * (rx_squared_error + rv_squared_error + rtheta_squared_error + rtheta_dot_squared_error) * 2)
        # distance =  -1 * (rx_squared_error + rv_squared_error + rtheta_squared_error + rtheta_dot_squared_error)

        return distance  # distance [0, inf) -> score [1, 0)

    def reference_tracking_error(self, states_real, states_reference):
        x_squared_error = (states_real[0] - states_reference[0]) ** 2 * self.params.x_error_weight
        v_squared_error = (states_real[1] - states_reference[1]) ** 2 * self.params.v_error_weight
        theta_squared_error = (states_real[2] - states_reference[2]) ** 2 * self.params.theta_error_weight
        theta_dot_squared_error = (states_real[3] - states_reference[3]) ** 2 * self.params.theta_dot_error_weight

        error = -1 * (x_squared_error + v_squared_error + theta_squared_error + theta_dot_squared_error)

        return error
