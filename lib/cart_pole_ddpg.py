from lib.agent.ddpg import DDPGParams, DDPGAgent
from lib.env.cart_pole import CartpoleParams, Cartpole
from lib.env.cart_pole import states2observations
from lib.logger.logger import LoggerParams, Logger, plot_trajectory
from lib.utils import ReplayMemory
import numpy as np
import copy


class Params:
    def __init__(self):
        self.cartpole_params = CartpoleParams()
        self.agent_params = DDPGParams()
        self.logger_params = LoggerParams()


class CartpoleDDPG:
    def __init__(self, params: Params):
        self.params = params
        self.cartpole = Cartpole(self.params.cartpole_params)
        self.shape_observations = self.cartpole.states_observations_dim
        self.shape_action = self.cartpole.action_dim
        self.replay_mem = ReplayMemory(self.params.agent_params.total_training_steps)

        if self.params.cartpole_params.observe_reference_states:
            self.shape_observations += self.cartpole.states_observations_refer_dim

        self.agent = DDPGAgent(self.params.agent_params,
                               shape_observations=self.shape_observations,
                               shape_action=self.shape_action,
                               model_path=self.params.agent_params.model_path,
                               mode=self.params.logger_params.mode)

        self.logger = Logger(self.params.logger_params)

    def interaction_step(self, mode=None):

        current_states = copy.deepcopy(self.cartpole.states)

        observations, _ = states2observations(current_states)

        states_refer_current = copy.deepcopy(self.cartpole.states_refer)

        if self.params.cartpole_params.observe_reference_states:
            observations_refer, _ = states2observations(states_refer_current)
            observations = observations.extend(observations_refer)

        action = self.agent.get_action(observations, mode)

        next_states = self.cartpole.step(action)

        if self.params.cartpole_params.update_reference_model:
            self.cartpole.refer_step()

        observations_next, failed = states2observations(next_states)
        r, distance_score = self.cartpole.reward_fcn(current_states, action, next_states, states_refer_current)

        return observations, action, observations_next, failed, r, distance_score

    def evaluation(self, reset_states=None, mode=None):

        if self.params.cartpole_params.random_reset_eval:
            self.cartpole.random_reset()
        else:
            self.cartpole.reset(reset_states)

        reward_list = []
        distance_score_list = []
        failed = False
        trajectory_tensor = []
        reference_trajectory_tensor = []

        for step in range(self.params.agent_params.max_episode_steps):

            observations, action, observations_next, failed, r, distance_score = \
                self.interaction_step(mode='eval')

            if self.params.logger_params.visualize_eval:
                self.cartpole.render()

            if mode == 'test':
                trajectory_tensor.append(self.cartpole.states[:4])
                reference_trajectory_tensor.append(self.cartpole.states_refer[:4])

            reward_list.append(r)
            distance_score_list.append(distance_score)

            if failed:
                break

        mean_reward = np.mean(reward_list)
        mean_distance_score = np.mean(distance_score_list)

        if mode == 'test':
            plot_trajectory(trajectory_tensor, reference_trajectory_tensor)

        return mean_reward, mean_distance_score, failed

    def train(self):
        ep = 0
        global_steps = 0
        best_dsas = 0.0  # Best distance score and survived
        moving_average_dsas = 0.0

        while global_steps < self.params.agent_params.total_training_steps:

            if self.params.cartpole_params.random_reset_train:
                self.cartpole.random_reset()
            else:
                self.cartpole.reset()

            ep += 1

            reward_list = []
            distance_score_list = []
            critic_loss_list = []

            failed = False

            for step in range(self.params.agent_params.max_episode_steps):

                observations, action, observations_next, failed, r, distance_score = \
                    self.interaction_step(mode='train')

                self.replay_mem.add((observations, action, r, observations_next, failed))

                reward_list.append(r)
                distance_score_list.append(distance_score)

                if self.replay_mem.get_size() > self.params.agent_params.experience_prefill_size:
                    minibatch = self.replay_mem.sample(self.params.agent_params.batch_size)
                    critic_loss = self.agent.optimize(minibatch)
                else:
                    critic_loss = 100

                critic_loss_list.append(critic_loss)
                global_steps += 1

                if failed:
                    break

            mean_reward = np.mean(reward_list)
            mean_distance_score = np.mean(distance_score_list)
            mean_critic_loss = np.mean(critic_loss_list)

            self.logger.log_training_data(mean_reward, mean_distance_score, mean_critic_loss, failed, global_steps)
            print(
                f"Training at {ep} episodes: average_reward: {mean_reward:.6}, distance_score: {mean_distance_score:.6}, "
                f"critic_loss: {mean_critic_loss:.6} ")

            if ep % self.params.logger_params.evaluation_period == 0:
                eval_mean_reward, eval_mean_distance_score, eval_failed = self.evaluation()
                self.logger.log_evaluation_data(eval_mean_reward, eval_mean_distance_score, eval_failed, global_steps)
                moving_average_dsas = 0.95 * moving_average_dsas + 0.05 * eval_mean_distance_score
                if moving_average_dsas > best_dsas:
                    self.agent.save_weights(self.logger.model_dir + '_best')
                    best_dsas = moving_average_dsas
            self.agent.save_weights(self.logger.model_dir)

    def test(self):
        self.evaluation(mode='test')
