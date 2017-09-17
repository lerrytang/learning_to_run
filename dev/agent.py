from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Lambda, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.initializers import VarianceScaling, RandomUniform
from keras.optimizers import Adam
import keras.backend as K
import numpy as np


def actor_loss(y_true, y_pred):
    return -K.mean(y_pred)


def zero_loss(y_true, y_pred):
    return K.mean(y_pred*0, axis=-1)

    
class DDPG:
    """
    DDPG Agent
    """
    
    def __init__(self, env, memory, rand_process, ob_processor, config):

        self.env = env
        self.ob_processor = ob_processor
        self.ob_dim = (self.env.observation_space.shape[0]+ob_processor.get_aug_dim(), )
        self.act_dim = self.env.action_space.shape
        self.act_high = self.env.action_space.high
        self.act_low = self.env.action_space.low
        self.memory = memory
        self.rand_process = rand_process

        # training strategy related
        self.num_train = config["num_train"] if "num_train" in config else 1
        self.save_prob = config["save_prob"] if "save_prob" in config else 1.0

        # learning related parameters
        self.batch_size = config["batch_size"] if "batch_size" in config else 64
        self.tau = config["tau"] if "tau" in config else 1e-3
        self.gamma = config["gamma"] if "gamma" in config else 0.99
        self.actor_l2 = config["actor_l2"] if "actor_l2" in config else 0
        self.critic_l2 = config["critic_l2"] if "critic_l2" in config else 1e-2
        self.actor_lr = config["actor_lr"] if "actor_lr" in config else 1e-3
        self.critic_lr = config["critic_lr"] if "critic_lr" in config else 1e-4
        self.merge_at_layer = config["merge_at_layer"] if "merge_at_layer" in config else 0

        # callbacks
        self.on_episode_end = []

    # ==================================================== #
    #           Building Models                            #
    # ==================================================== #

    def build_nets(self, actor_hiddens=[400, 300], scale_action=None, critic_hiddens=[400, 300], lrelu=-1):
        """
        scale_action is either None (leave the action to be [-1, 1],
        or a function (Lambda layer) that takes action tensor as input and return a scaled action
        """

        # build models
        self.actor, self.num_alayers = self.create_actor(actor_hiddens, scale_action, critic_hiddens, lrelu)
        self.target_actor, _ = self.create_actor(actor_hiddens, scale_action, critic_hiddens, lrelu, trainable=False)
        self.critic, self.num_clayers = self.create_critic(critic_hiddens, lrelu)

        # hard copy weights
        self._copy_critic_weights(self.critic, self.actor)
        self._copy_critic_weights(self.critic, self.target_actor)
        self._copy_actor_weights(self.actor, self.target_actor)

    def _build_critic_part(self, ob_input, act_input, critic_hiddens, lrelu, trainable=True):

        assert self.merge_at_layer<=len(critic_hiddens)

        # critic input part
        if self.merge_at_layer == 0:
            x = Concatenate(name="combined_input")([ob_input, act_input])
        else:
            x = ob_input

        # critic hidden part
        for i, num_hiddens in enumerate(critic_hiddens):
            x = Dense(num_hiddens, activation=None, trainable=trainable,
                    kernel_initializer=VarianceScaling(scale=1.0/3, distribution="uniform"),
                    bias_initializer=VarianceScaling(scale=1.0/3, distribution="uniform"),
                    kernel_regularizer=l2(self.critic_l2), name="critic_fc{}".format(i+1))(x)
            if lrelu>0:
                x = LeakyReLU(name="critic_lrelu{}".format(i+1))(x)
            else:
                x = Activation("relu", name="critic_relu{}".format(i+1))(x)
            if self.merge_at_layer == i+1:
                x = Concatenate(name="combined_input")([x, act_input])

        # critic output
        qval = Dense(1, activation="linear", trainable=trainable,
                     kernel_initializer=RandomUniform(minval=-3e-4, maxval=3e-4),
                     bias_initializer=RandomUniform(minval=-3e-4, maxval=3e-4),
                     kernel_regularizer=l2(self.critic_l2), name="qval")(x)
        return qval
    
    def create_actor(self, actor_hiddens, scale_action, critic_hiddens, lrelu, trainable=True):
        # actor input part
        ob_input = Input(shape=self.ob_dim, name="ob_input")
        x = ob_input
        
        # actor hidden part
        for i, num_hiddens in enumerate(actor_hiddens):
            x = Dense(num_hiddens, activation=None, trainable=trainable,
                    kernel_initializer=VarianceScaling(scale=1.0/3, distribution="uniform"),
                    bias_initializer=VarianceScaling(scale=1.0/3, distribution="uniform"),
                    kernel_regularizer=l2(self.actor_l2), name="actor_fc{}".format(i+1))(x)
            if lrelu>0:
                x = LeakyReLU(name="actor_lrelu{}".format(i+1))(x)
            else:
                x = Activation("relu", name="actor_relu{}".format(i+1))(x)

        # action output
        x = Dense(self.act_dim[0], activation="tanh", trainable=trainable,
                  kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3),
                  bias_initializer=RandomUniform(minval=-3e-3, maxval=3e-3),
                       kernel_regularizer=l2(self.actor_l2), name="action")(x)
        if scale_action is not None:
            action = scale_action(x)
        else:
            action = x
        
        # untrainable critic part
        qval = self._build_critic_part(ob_input, action, critic_hiddens, lrelu, trainable=False)
        
        # compile model
        actor = Model(inputs=[ob_input], outputs=[action, qval])
        optimizer = Adam(lr=self.actor_lr)
        actor.compile(optimizer=optimizer, loss=[zero_loss, actor_loss], loss_weights=[0, 1])
        return actor, len(actor_hiddens)
    
    def create_critic(self, critic_hiddens, lrelu, trainable=True):
        # critic input part
        ob_input = Input(shape=self.ob_dim, name="ob_input")
        act_input = Input(shape=self.act_dim, name="act_input")

        # critic part
        qval = self._build_critic_part(ob_input, act_input, critic_hiddens, lrelu, trainable=trainable)

        # compile
        critic = Model(inputs=[ob_input, act_input], outputs=[qval])
        optimizer = Adam(lr=self.critic_lr)
        critic.compile(optimizer=optimizer, loss="mse")
        return critic, len(critic_hiddens)
        
    # ==================================================== #
    #           Network Weights Copy                       #
    # ==================================================== #

    def _copy_layer_weights(self, src_layer, tar_layer, tau=1.0):
        src_weights = src_layer.get_weights()
        tar_weights = tar_layer.get_weights()
        assert len(src_weights) == len(tar_weights)
        for i in xrange(len(src_weights)):
                tar_weights[i] = tau * src_weights[i] + (1.0 - tau) * tar_weights[i]
        tar_layer.set_weights(tar_weights)
    
    def _copy_actor_weights(self, src_model, tar_model, tau=1.0):
        actor_layers = ["actor_fc" + str(i+1) for i in xrange(self.num_alayers)] + ["action"]
        for l in actor_layers:
            src_layer = src_model.get_layer(l)
            tar_layer = tar_model.get_layer(l)
            self._copy_layer_weights(src_layer, tar_layer, tau)
    
    def _copy_critic_weights(self, src_model, tar_model, tau=1.0):
        critic_layers = ["critic_fc" + str(i+1) for i in xrange(self.num_clayers)] + ["qval"]
        for l in critic_layers:
            src_layer = src_model.get_layer(l)
            tar_layer = tar_model.get_layer(l)
            self._copy_layer_weights(src_layer, tar_layer, tau)
    
    # ==================================================== #
    #          Traing Models                               #
    # ==================================================== #
    
    def _train_critic(self, ob0, action, reward, ob1, done):
        future_action, future_q = self.target_actor.predict_on_batch(ob1)
        future_q = future_q.squeeze()
        reward += self.gamma * future_q * (1 - done)
        hist = self.critic.fit([ob0, action], reward,
                batch_size=self.batch_size, verbose=0)
        self._copy_critic_weights(self.critic, self.actor)
        return hist
    
    def _train_actor(self, ob0, action, reward, ob1, done):
        # the output signals are just dummy
        hist = self.actor.fit([ob0], [reward, reward],
                batch_size=self.batch_size, verbose=0)
        return hist
    
    def train_actor_critic(self):
        ob0, action, reward, ob1, done = self.memory.sample(self.batch_size)
        if ob0 is None:
            return 0, 0
        else:
            # train critic
            critic_hist = self._train_critic(ob0, action, reward, ob1, done)
            # DEBUG
            aa, q_actor = self.actor.predict_on_batch([ob0])
            q_critic = self.critic.predict_on_batch([ob0, aa])
            assert np.allclose(q_actor, q_critic)
            # train actor
            actor_hist = self._train_actor(ob0, action, reward, ob1, done)
            # soft update weights
            self._copy_critic_weights(self.critic, self.target_actor, tau=self.tau)
            self._copy_actor_weights(self.actor, self.target_actor, tau=self.tau)
            
            return critic_hist.history["loss"][0], -1*actor_hist.history["qval_loss"][0]
    
    # ==================================================== #
    #                   Trial Logic                        #
    # ==================================================== #

    def append_hist(self, hist, data):
        if hist is None:
            hist = np.copy(data)
        else:
            hist = np.vstack([hist, data])
        return hist
    
    def learn(self, max_steps=1000, total_episodes=10000):
        episode_n = 0
        episode_reward = 0
        episode_steps = 0
        episode_losses = []
        episode_qval = []
        action_hist = None
        noisy_action_hist = None
        observation_hist = None
        reward_hist = []
        steps_hist = []
        new_ob = self.env.reset()
        self.ob_processor.reset()
        zero_action = np.zeros(self.env.action_space.shape)
        first_frame = True
        done = False

        while episode_n<total_episodes:

            # ignore first frame because it contains phantom obstacle
            if not done and first_frame:
                new_ob, reward, done, info = self.env.step(zero_action)
                episode_reward += reward
                episode_steps += 1
                first_frame = False
                assert not done, "Episode finished in one step"
                continue

            # select and execute action
            new_ob = self.ob_processor.process(new_ob)
            observation = np.reshape(new_ob, [1, -1])
            observation_hist = self.append_hist(observation_hist, observation)
            action, _ = self.actor.predict(observation)
            action_hist = self.append_hist(action_hist, action)
            action += self.rand_process.sample()
            action = np.clip(action, self.act_low, self.act_high)
            noisy_action_hist = self.append_hist(noisy_action_hist, action)
            new_ob, reward, done, info = self.env.step(action.squeeze())

            # bookkeeping
            episode_reward += reward
            episode_steps += 1

            # store experience
            if np.random.rand() < self.save_prob:
                assert np.all((action>=self.act_low) & (action<=self.act_high))
                self.memory.store(observation, action, reward, done)

            # train
            for _ in xrange(self.num_train):
                loss, qval = self.train_actor_critic()
                if loss is not None:
                    episode_losses.append(loss)
                if qval is not None:
                    episode_qval.append(qval)

            # on episode end
            if done or episode_steps>=max_steps:
                episode_n += 1
                reward_hist.append(episode_reward)
                steps_hist.append(episode_steps)

                # call callbacks
                episode_info = {
                        "episode": episode_n,
                        "steps": episode_steps,
                        "total_reward": episode_reward,
                        "loss": episode_losses,
                        "qval": episode_qval,
                        "agent": self}
                for func in self.on_episode_end:
                    func(episode_info)

                # reset values
                episode_reward = 0
                episode_steps = 0
                episode_losses = []
                episode_qval = []
                action_hist = None
                noisy_action_hist = None
                observation_hist = None
                new_ob = self.env.reset()
                self.ob_processor.reset()
                first_frame = True
                done = False

        return reward_hist, steps_hist

    def test(self, max_steps):
        episode_reward = 0
        episode_steps = 0
        new_ob = self.env.reset()
        self.ob_processor.reset()
        zero_action = np.zeros(self.env.action_space.shape)
        first_frame = True
        done = False
        while not done:

            # ignore first frame because it contains phantom obstacle
            if first_frame:
                new_ob, reward, done, info = self.env.step(zero_action)
                episode_reward += reward
                episode_steps += 1
                first_frame = False
                assert not done, "Episode finished in one step"
                continue

            new_ob = self.ob_processor.process(new_ob)
            observation = np.reshape(new_ob, [1, -1])
            action, _ = self.actor.predict(observation)
            action = np.clip(action, self.act_low, self.act_high)
            new_ob, reward, done, info = self.env.step(action.squeeze())
            episode_reward += reward
            episode_steps += 1
            done = done | (episode_steps>=max_steps)
        return episode_steps, episode_reward
        
    def save_models(self, paths):
        self.actor.save_weights(paths["actor"])
        self.critic.save_weights(paths["critic"])
        self.target_actor.save_weights(paths["target"])

    def load_models(self, paths):
        self.actor.load_weights(paths["actor"])
        self.critic.load_weights(paths["critic"])
        self.target_actor.load_weights(paths["target"])

    def save_memory(self, path):
        self.memory.save_memory(path)

    def load_memory(self, path):
        self.memory.load_memory(path)
