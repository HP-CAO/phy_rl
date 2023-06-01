import math

from lib.agent.ddpg import DDPGParams, DDPGAgent
from lib.agent.network import TaylorParams
from lib.logger.logger import LoggerParams, Logger, plot_trajectory
from lib.utils import ReplayMemory
from lib.env.locomotion.envs.a1_env import A1Params, A1Robot
import matplotlib.pyplot as plt

import numpy as np
import copy


class Params:
    def __init__(self):
        self.a1_params = A1Params()
        self.agent_params = DDPGParams()
        self.logger_params = LoggerParams()
        self.taylor_params = TaylorParams()


class A1DDPG:
    def __init__(self, params: Params):
        self.params = params

        self.a1 = A1Robot(self.params.a1_params)
        self.shape_observations = self.a1.states_observations_dim
        self.shape_action = self.a1.action_dim
        self.replay_mem = ReplayMemory(self.params.agent_params.replay_buffer_size)

        self.agent = DDPGAgent(self.params.agent_params,
                               self.params.taylor_params,
                               shape_observations=self.shape_observations,
                               shape_action=self.shape_action,
                               model_path=self.params.agent_params.model_path,
                               mode=self.params.logger_params.mode)

        self.logger = Logger(self.params.logger_params)

    def interaction_step(self, mode=None):

        # observations = copy.deepcopy(self.a1.observation)
        observations = copy.deepcopy(self.a1.get_tracking_error())

        action = self.agent.get_action(observations, mode)

        _, terminal, abort = self.a1.step(action, action_mode=self.params.agent_params.action_mode)

        observations_next = self.a1.get_tracking_error()

        r = self.a1.get_reward()
        # print("reward:", r)
        return observations, action, observations_next, terminal, r, abort

    def evaluation(self, reset_states=None, mode=None):

        if self.params.a1_params.random_reset_eval:
            self.a1.random_reset()
        else:
            self.a1.reset(reset_states)

        reward_list = []
        failed = False

        for step in range(self.params.agent_params.max_episode_steps):

            observations, action, observations_next, failed, r, abort = \
                self.interaction_step(mode='eval')

            if abort:
                break

            reward_list.append(r)

            if failed:
                break

        if len(reward_list) == 0:
            mean_reward = math.nan
            mean_distance_score = math.nan
        else:
            mean_reward = np.mean(reward_list)
            # mean_distance_score = np.mean(distance_score_list)
            mean_distance_score = 0

        if mode == 'test':
            # np.save("drl_trajectory", self.a1.states_vector)
            # plt.figure(figsize=(9, 6))
            # plt.plot(np.arange(len(self.a1.states_vector)), self.a1.states_vector, label='vx')
            # plt.legend(loc='best')
            # plt.grid(True)
            # plt.tight_layout()
            # plt.savefig("trajectory.png", dpi=300)
            np.save("height_trajectory", self.a1.height_vector)
            plt.figure(figsize=(9, 6))
            plt.plot(np.arange(len(self.a1.height_vector)), self.a1.height_vector, label='vx')
            plt.legend(loc='best')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("height_trajectory.png", dpi=300)

        eval_steps = len(reward_list)
        return mean_reward, mean_distance_score, failed, eval_steps

    def train(self):
        ep = 0
        global_steps = 0
        best_dsas = -10  # Best distance score and survived
        moving_average_reward = 0.0

        while global_steps < self.params.agent_params.total_training_steps:

            if self.params.a1_params.random_reset_train:
                self.a1.random_reset()
            else:
                self.a1.reset(step=global_steps)

            ep += 1
            reward_list = []
            critic_loss_list = []

            failed = False

            ep_steps = 0

            for step in range(self.params.agent_params.max_episode_steps):

                observations, action, observations_next, failed, r, abort = \
                    self.interaction_step(mode='train')

                if abort:
                    break

                self.replay_mem.add((observations, action, r, observations_next, failed))

                reward_list.append(r)

                if self.replay_mem.get_size() > self.params.agent_params.experience_prefill_size:
                    minibatch = self.replay_mem.sample(self.params.agent_params.batch_size)
                    critic_loss = self.agent.optimize(minibatch)
                else:
                    critic_loss = 100

                critic_loss_list.append(critic_loss)
                global_steps += 1
                ep_steps += 1

                if failed:
                    break

            if len(reward_list) == 0:
                continue
            else:
                mean_reward = np.mean(reward_list)
                mean_critic_loss = np.mean(critic_loss_list)

            self.logger.log_training_data(mean_reward, 0, mean_critic_loss, failed, global_steps)
            print(f"Training at {ep} episodes: average_reward: {mean_reward:.6},"
                  f"critic_loss: {mean_critic_loss:.6}, total_steps_ep: {ep_steps} ")

            if ep % self.params.logger_params.evaluation_period == 0:
                eval_mean_reward, eval_mean_distance_score, eval_failed, eval_steps = self.evaluation()
                self.logger.log_evaluation_data(eval_mean_reward, eval_mean_distance_score, eval_failed, global_steps)
                moving_average_reward = 0.95 * moving_average_reward + 0.05 * eval_mean_reward
                if moving_average_reward > best_dsas:
                    self.agent.save_weights(self.logger.model_dir + '_best')
                    best_dsas = moving_average_reward
                if eval_steps == 10000:
                    self.agent.save_weights(self.logger.model_dir + f'_{global_steps}')
            self.agent.save_weights(self.logger.model_dir)

    def test(self):
        self.evaluation(mode='test')
