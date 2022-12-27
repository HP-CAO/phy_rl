import os
import tensorflow as tf
from lib.utils import OrnsteinUhlenbeckActionNoise
from lib.agent.base import BaseAgent, BaseAgentParams


class DDPGAgentParams(BaseAgentParams):
    def __init__(self):
        super(DDPGAgentParams, self).__init__()


class DDPGAgent(BaseAgent):

    def __init__(self, params: DDPGAgentParams, shape_observations=5, shape_targets=2, shape_action=1, on_edge=False):
        super(DDPGAgent, self).__init__(params, shape_observations, shape_targets, shape_action, on_edge)
        self.params = params
        self.shape_observations = shape_observations
        self.shape_targets = shape_targets
        self.shape_action = shape_action
        self.on_edge = on_edge
        self.action_noise = OrnsteinUhlenbeckActionNoise(self.shape_action)
        self.action_noise_factor = params.action_noise_factor
        self.add_actions_observations = self.params.add_actions_observations
        self.action_observations_dim = self.params.action_observations_dim

    def initial_model(self):
        self.actor = self.build_actor("normal-")
        if not self.on_edge:
            self.critic = self.build_critic("normal-")
            self.critic_target = self.build_critic("target-")
            self.actor_target = self.build_actor("target-")
            self.hard_update()

    def hard_update(self):
        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target.set_weights(self.critic.get_weights())

    def soft_update(self):
        soft_alpha = tf.convert_to_tensor(self.params.soft_alpha, dtype=tf.float32)
        self._soft_update(soft_alpha)

    @tf.function
    def _soft_update(self, soft_alpha):
        # Obtain weights directly as tf.Variables
        actor_weights = self.actor.weights
        actor_target_weights = self.actor_target.weights
        critic_weights = self.critic.weights
        critic_target_weights = self.critic_target.weights

        for w_new, w_old in zip(actor_weights, actor_target_weights):
            w_old.assign(w_new * soft_alpha + w_old * (1. - soft_alpha))

        for w_new, w_old in zip(critic_weights, critic_target_weights):
            w_old.assign(w_new * soft_alpha + w_old * (1. - soft_alpha))

    def save_weights(self, model_name):
        self.actor.save_weights('./models/' + model_name + '_DDPG/actor_weights')
        self.critic.save_weights('./models/' + model_name + '_DDPG/critic_weights')
        self.actor_target.save_weights('./models/' + model_name + '_DDPG/actor_target_weights')
        self.critic_target.save_weights('./models/' + model_name + '_DDPG/critic_target_weights')

    def load_weights(self, path_to_weights):
        print("loading pretrained weights......")

        if not os.path.exists(path_to_weights):
            raise IOError("Weights path not exist")

        path_to_actor = path_to_weights + 'actor_weights'
        path_to_critic = path_to_weights + 'critic_weights'
        path_to_actor_target = path_to_weights + 'actor_target_weights'
        path_to_critc_target = path_to_weights + 'critic_target_weights'

        self.actor.load_weights(path_to_actor)
        self.critic.load_weights(path_to_critic)
        self.actor_target.load_weights(path_to_actor_target)
        self.critic_target.load_weights(path_to_critc_target)
