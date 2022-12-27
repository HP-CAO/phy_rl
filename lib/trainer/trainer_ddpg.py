import threading
import tensorflow as tf
from lib.trainer.replay_mem import ReplayMemory
from lib.agent.ddpg import DDPGAgent
from lib.trainer.trainer_params import OffPolicyTrainerParams


class DDPGTrainerParams(OffPolicyTrainerParams):
    def __init__(self):
        super().__init__()
        self.actor_update_period = 1


class DDPGTrainer:
    def __init__(self, params: DDPGTrainerParams, agent: DDPGAgent, *args, **kwargs):
        self.params = params
        self.agent = agent
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=self.params.learning_rate_critic)
        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=self.params.learning_rate_actor)
        self.replay_mem = ReplayMemory(size=self.params.rm_size,
                                       combined_experience_replay=self.params.combined_experience_replay)
        self.replay_memory_mutex = threading.Lock()
        self.critic_update = tf.Variable(0, dtype=tf.int64)

    def store_experience(self, observations, targets, action, reward, next_observations, failed):
        if self.params.is_remote_train:
            self.replay_memory_mutex.acquire()
            self.replay_mem.add((observations, targets, action, reward, next_observations, failed))
            self.replay_memory_mutex.release()
        else:
            self.replay_mem.add((observations, targets, action, reward, next_observations, failed))

    def optimize(self):

        # for i in range(self.params.training_epoch):

        if self.params.pre_fill_exp > self.replay_mem.get_size():
            return 0

        self.replay_memory_mutex.acquire()
        mini_batch = self.replay_mem.sample(self.params.batch_size)
        self.replay_memory_mutex.release()

        # Convert to tensor for using @tf.function
        ob1_tf = tf.convert_to_tensor(mini_batch[0], dtype=tf.float32)
        tgs_tf = tf.convert_to_tensor(mini_batch[1], dtype=tf.float32)
        a1_tf = tf.convert_to_tensor(mini_batch[2], dtype=tf.float32)
        r1_tf = tf.convert_to_tensor(mini_batch[3], dtype=tf.float32)
        ob2_tf = tf.convert_to_tensor(mini_batch[4], dtype=tf.float32)
        cra_tf = tf.convert_to_tensor(mini_batch[5], dtype=tf.float32)
        replay_memory_size_tf = tf.convert_to_tensor(self.replay_mem.get_size(), dtype=tf.float32)

        # ------------------- optimize critic and actor ----------------
        # Use target actor exploitation policy here for loss evaluation
        # Use graph mode for tf
        loss_critic = self._optimize(ob1_tf, tgs_tf, a1_tf, r1_tf, ob2_tf, cra_tf, replay_memory_size_tf)
        self.agent.soft_update()

        return loss_critic.numpy()

    @tf.function
    def _optimize(self, ob1, tgs, a1, r1, ob2, cra, replay_memory_size):

        # ---------------------- optimize critic ----------------------
        with tf.GradientTape() as tape:

            a2 = self.agent.actor_target([ob2, tgs])

            if self.params.target_action_noise:
                action_noise = tf.clip_by_value(
                    tf.random.normal(shape=(self.params.batch_size, 1), mean=0, stddev=0.3),
                    clip_value_min=-0.5, clip_value_max=0.5)
                a2 = tf.clip_by_value((a2 + action_noise), clip_value_min=-1, clip_value_max=1)

            q_e = self.agent.critic_target([ob2, tgs, a2])

            y_exp = r1 + self.params.gamma_discount * q_e * (1 - cra)
            y_pre = self.agent.critic([ob1, tgs, a1])

            loss_critic = tf.keras.losses.mean_squared_error(y_exp, y_pre)

        q_grads = tape.gradient(loss_critic, self.agent.critic.trainable_variables)
        self.optimizer_critic.apply_gradients(zip(q_grads, self.agent.critic.trainable_variables))


        # ---------------------- optimize actor ----------------------

        self.critic_update.assign_add(1)

        # Does not case tracing due to use of tensors for if statement
        if self.critic_update % self.params.actor_update_period == 0:
            # IMPORTANT: Do not use replay.get_size() directly with tf.function, will be static!!
            if replay_memory_size >= self.params.actor_freeze_step_count:
                with tf.GradientTape() as tape:
                    a1_predict = self.agent.actor([ob1, tgs])
                    actor_value = -1 * tf.math.reduce_mean(self.agent.critic([ob1, tgs, a1_predict]))
                actor_gradients = tape.gradient(actor_value, self.agent.actor.trainable_variables)
                self.optimizer_actor.apply_gradients(zip(actor_gradients, self.agent.actor.trainable_variables))

        return tf.reduce_mean(loss_critic)

    def load_weights(self, path):
        self.agent.load_weights(path)
