import numpy as np
import copy
import math
from lib.utils import states2observations
from lib.monitor.monitor import ModelStatsParams, ModelStats
from lib.env.gym_physics import GymPhysics, GymPhysicsParams
from lib.env.linear_physics import LinearPhysicsParams, LinearPhysics
from lib.env.reward import RewardParams, RewardFcn


class ModelTrackParams:
    def __init__(self):
        self.physics_params = GymPhysicsParams()
        self.linear_physics_params = LinearPhysicsParams()
        self.reward_params = RewardParams()
        self.stats_params = ModelStatsParams()
        self.agent_params = None


class ModelTrackSystem:
    def __init__(self, params: ModelTrackParams):
        self.params = params

        self.physics = GymPhysics(self.params.physics_params)
        self.reference_model = LinearPhysics(self.params.linear_physics_params)
        self.model_stats = ModelStats(self.params.stats_params, self.physics)
        self.reward_fcn = RewardFcn(self.params.reward_params)
        self.shape_targets = self.model_stats.get_shape_targets()
        self.shape_observations = self.physics.get_shape_observations()
        self.trainer = None
        self.agent = None

    def evaluation_episode(self, ep, agent, reset_states=None):
        self.model_stats.init_episode()
        reset_states = [0., 0., 2 * -math.pi/180, 0., False]
        self.physics.reset(reset_states)
        self.reference_model.reset(reset_states)

        if agent.add_actions_observations:
            action_observations = np.zeros(shape=agent.action_observations_dim)
        else:
            action_observations = []

        for step in range(self.params.stats_params.max_episode_steps):

            if self.params.stats_params.visualize_eval:
                self.physics.render()
                self.reference_model.render()

            observations = np.hstack((self.model_stats.observations, action_observations))

            action = agent.get_exploitation_action(observations, self.model_stats.targets)

            if self.params.agent_params.add_actions_observations:
                action_observations = np.append(action_observations, action)[1:]

            states_next = self.physics.step(action)
            refer_states = self.reference_model.step()

            stats_observations_next, failed = states2observations(states_next)

            performance_reward = self.reward_fcn.reward(
                self.model_stats.observations, self.model_stats.targets, action, failed,
                pole_length=self.params.physics_params.length)

            tracking_reward = self.reward_fcn.reference_tracking_error(states_next, refer_states)
            r = self.params.reward_params.tracking_error_weight * tracking_reward + performance_reward

            self.model_stats.observations = copy.deepcopy(stats_observations_next)
            self.model_stats.measure(self.model_stats.observations, self.model_stats.targets,
                                     failed, pole_length=self.params.physics_params.length,
                                     distance_score_factor=self.params.reward_params.distance_score_factor)

            self.model_stats.reward.append(r)
            self.model_stats.actions_std.append(self.physics.actions_std)
            self.model_stats.cart_positions.append(self.physics.states[0])
            self.model_stats.pendulum_angele.append(self.physics.states[2])
            self.model_stats.actions.append(action)

            # if failed:
            #     break

        if not self.model_stats.params.eval_on_multi_conditions:
            self.model_stats.evaluation_monitor_scalar(ep)
            self.model_stats.evaluation_monitor_image(ep)

        distance_score_and_survived = float(self.model_stats.survived) * self.model_stats.get_average_distance_score()

        self.physics.close()

        return distance_score_and_survived

    def train(self):

        ep = 0
        global_steps = 0
        best_dsas = 0.0  # Best distance score and survived
        moving_average_dsas = 0.0
        while self.model_stats.total_steps < self.model_stats.params.total_steps:

            self.model_stats.init_episode()
            self.reference_model.reset(self.physics.states)

            ep += 1
            step = 0

            if self.params.agent_params.add_actions_observations:
                action_observations = np.zeros(shape=self.params.agent_params.action_observations_dim)
            else:
                action_observations = []

            for step in range(self.params.stats_params.max_episode_steps):
                observations = np.hstack((self.model_stats.observations, action_observations)).tolist()

                action = self.agent.get_exploration_action(observations, self.model_stats.targets)

                if self.params.agent_params.add_actions_observations:
                    action_observations = np.append(action_observations, action)[1:]

                states_next = self.physics.step(action)  # real_states
                refer_states = self.reference_model.step()

                stats_observations_next, failed = states2observations(states_next)
                observations_next = np.hstack((stats_observations_next, action_observations)).tolist()

                performance_reward = self.reward_fcn.reward(
                    self.model_stats.observations, self.model_stats.targets, action, failed,
                    pole_length=self.params.physics_params.length)

                tracking_reward = self.reward_fcn.reference_tracking_error(states_next, refer_states)

                r = self.params.reward_params.tracking_error_weight * tracking_reward + performance_reward

                self.trainer.store_experience(observations, self.model_stats.targets, action, r, observations_next, failed)

                self.model_stats.observations = copy.deepcopy(stats_observations_next)

                self.model_stats.measure(self.model_stats.observations, self.model_stats.targets, failed,
                                         pole_length=self.params.physics_params.length,
                                         distance_score_factor=self.params.reward_params.distance_score_factor)

                self.model_stats.reward.append(r)
                self.model_stats.actions_std.append(self.physics.actions_std)

                critic_loss = self.trainer.optimize()
                self.model_stats.add_critic_loss(critic_loss)

                global_steps += 1

                # this is for balancing the experiences
                # if self.model_stats.consecutive_on_target_steps > self.params.stats_params.on_target_reset_steps:
                #     break
                #
                # if failed:
                #     break
            self.model_stats.add_steps(step)
            self.model_stats.training_monitor(ep)
            self.agent.noise_factor_decay(self.model_stats.total_steps)

            if ep % self.params.stats_params.eval_period == 0:
                dsal = self.multi_episodes_evaluation(ep)
                moving_average_dsas = 0.95 * moving_average_dsas + 0.05 * dsal
                if moving_average_dsas > best_dsas:
                    self.agent.save_weights(self.params.stats_params.model_name + '_best')
                    best_dsas = moving_average_dsas

        self.agent.save_weights(self.params.stats_params.model_name)

    def multi_episodes_evaluation(self, ep):
        """Here we evaluate the performance of the agent on different conditions"""
        multi_conditions = self.physics.multi_eval_conditions
        score_list = []
        episode_log_data_list = []
        print("Evaluating......")

        for cond in multi_conditions:
            distance_score_and_survived = self.evaluation_episode(ep, self.agent, cond)
            score_list.append(distance_score_and_survived)
            episode_log_data = self.model_stats.log_data()
            episode_log_data_list.append(episode_log_data)

        self.model_stats.multi_evaluation_scalar(ep, episode_log_data_list)
        return np.mean(score_list)
