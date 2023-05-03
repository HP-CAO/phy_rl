from lib.agent.ddpg import DDPGParams, DDPGAgent, TaylorParams
from lib.env.a1 import A1Robot, A1Params
from lib.logger.logger import LoggerParams, Logger
from lib.utils import ReplayMemory
import numpy as np


class Params:
    def __init__(self):
        self.a1_params = A1Params()
        self.agent_params = DDPGParams()
        self.logger_params = LoggerParams()


class A1DDPG:
    def __init__(self, params: Params):
        self.params = params

        self.robot = A1Robot(self.params.a1_params)

        self.shape_observations = self.robot.states_observations_dim()
        self.shape_action = self.robot.action_dim
        self.replay_mem = ReplayMemory(self.params.agent_params.total_training_steps)

        if self.params.a1_params.observe_reference_states:
            self.shape_observations += self.robot.states_observations_refer_dim

        self.agent = DDPGAgent(self.params.agent_params,
                               shape_observations=self.shape_observations,
                               shape_action=self.shape_action,
                               model_path=self.params.agent_params.model_path,
                               mode=self.params.logger_params.mode)

        self.logger = Logger(self.params.logger_params)

    def interaction_step(self, mode=None):

        observations = self.robot.getExtendedObservation()

        action = self.agent.get_action(observations, mode)

        observations_next, reward, failed, _ = \
            self.robot.step(action, use_residual=self.params.agent_params.as_residual_policy)

        return observations, action, observations_next, failed, reward

    def evaluation(self, reset_states=None, mode=None):

        if mode == "test":
            vis = True
        else:
            vis = False

        if self.params.a1_params.random_reset_eval:
            self.robot.random_reset()
        else:
            self.robot.reset(reset_states, vis=vis)

        reward_list = []

        for step in range(self.params.agent_params.max_episode_steps):

            observations, action, observations_next, failed, r = self.interaction_step(mode='eval')

            reward_list.append(r)

            if failed:
                break

        mean_reward = np.mean(reward_list)

        return mean_reward

    def train(self):
        ep = 0
        global_steps = 0
        best_dsas = -100.  # Best distance score and survived
        moving_average_reward = 0.0

        while global_steps < self.params.agent_params.total_training_steps:

            if self.params.a1_params.random_reset_train:
                self.robot.random_reset()
            else:
                self.robot.reset()

            ep += 1

            reward_list = []
            critic_loss_list = []

            failed = False
            ep_steps = 0

            for step in range(self.params.agent_params.max_episode_steps):

                observations, action, observations_next, failed, r = self.interaction_step(mode='train')

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

            mean_reward = np.mean(reward_list)
            mean_critic_loss = np.mean(critic_loss_list)

            self.logger.log_training_data(mean_reward, 0, mean_critic_loss, failed, global_steps)

            print(f"Training at {ep} episodes: average_reward: {mean_reward:.6},"
                  f"critic_loss: {mean_critic_loss:.6}, total_steps_ep: {ep_steps} ")

            if ep % self.params.logger_params.evaluation_period == 0:
                eval_mean_reward = self.evaluation()
                self.logger.log_evaluation_data(eval_mean_reward, 0, False, global_steps)
                moving_average_reward = 0.95 * moving_average_reward + 0.05 * eval_mean_reward
                if moving_average_reward > best_dsas:
                    self.agent.save_weights(self.logger.model_dir + '_best')
                    best_dsas = moving_average_reward
            self.agent.save_weights(self.logger.model_dir)

    def test(self):
        self.evaluation(mode='test', reset_states=None)
