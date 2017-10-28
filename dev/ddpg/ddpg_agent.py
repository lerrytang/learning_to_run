from agent import Agent
from nipsenv import NIPS
from env_sampler import EnvSampler
from multiprocessing import Queue, Process
from mem import ReplayBuffer as RB

from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Lambda, Activation, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.initializers import VarianceScaling, RandomUniform
from keras.optimizers import Adam
import keras.backend as K
from .layer_norm import LayerNorm

import numpy as np
import os
import util


def minus_Q(y_true, y_pred):
    return -K.mean(y_pred)


def action_sqr(y_true, y_pred):
    return K.mean(K.sum(K.square(y_pred), axis=-1))


def swish(x, name):
    y = Activation("sigmoid", name=name+"_sigmoid")(x)
    y = Lambda(lambda t: t*y, name=name)(x)
    return y


class DDPG(Agent):
    """
    DDPG Agent
    """

    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.logger = config["logger"]

        # backward compatibility
        if "use_ln" not in self.config:
            self.config["use_ln"] = False
        if "fake_ob_pos" not in self.config:
            self.config["fake_ob_pos"] = 0.0
        if "clear_vel" not in self.config:
            self.config["clear_vel"] = False
        if "include_limb_vel" not in self.config:
            self.config["include_limb_vel"] = True
        if "ob_dist_scale" not in self.config:
            self.config["ob_dist_scale"] = 1.0
        if "use_swish" not in self.config:
            self.config["use_swish"] = False
        if "num_samplers" not in self.config:
            self.config["num_samplers"] = 0

        if self.config["num_samplers"] > 0:
            self.logger.info("<Multiprocess>")
        else:
            self.logger.info("<Single process>")

        # observation processor
        self.ob_processor = util.create_ob_processor(env, config)
        self.ob_dim = \
            (self.env.observation_space.shape[0] + self.ob_processor.get_aug_dim(),)
        if "jump" in config and config["jump"]:
            assert self.env.action_space.shape[0] % 2 == 0
            real_act_dim = self.env.action_space.shape[0] / 2
            self.jump = True
        else:
            real_act_dim = self.env.action_space.shape[0]
            self.jump = False
        self.act_dim = (real_act_dim,)
        self.act_high = self.env.action_space.high[:real_act_dim]
        self.act_low = self.env.action_space.low[:real_act_dim]

        # random process for exploration
        self.rand_process = util.create_rand_process(env, config)

        # replay buffer
        self.memory = RB(
            ob_dim=self.ob_dim,
            act_dim=self.act_dim,
            capacity=config["memory_capacity"])

        # build actor, critic and target
        self.build_nets(
            actor_hiddens=config["actor_hiddens"],
            critic_hiddens=config["critic_hiddens"],
            lrelu=config["lrelu"])
        self.actor.summary()
        self.critic.summary()
        self.target.summary()

        self.log_dir = config["log_dir"]
        self.model_dir = config["model_dir"] if "model_dir" in config else self.log_dir

    # ==================================================== #
    #           Building Models                            #
    # ==================================================== #

    def build_nets(self, actor_hiddens, critic_hiddens, lrelu):

        # build models
        self.actor = self.create_actor(actor_hiddens, critic_hiddens, lrelu)
        self.target = self.create_actor(actor_hiddens, critic_hiddens, lrelu, trainable=False)
        self.critic = self.create_critic(critic_hiddens, lrelu)

        # hard copy weights
        self._copy_critic_weights(self.critic, self.actor)
        self._copy_critic_weights(self.critic, self.target)
        self._copy_actor_weights(self.actor, self.target)

    def _build_critic_part(self, ob_input, act_input, critic_hiddens, lrelu, trainable=True):

        assert self.config["merge_at_layer"] <= len(critic_hiddens)

        # critic input part
        if self.config["use_bn"]:
            x = BatchNormalization(center=False,
                                   scale=False,
                                   trainable=trainable,
                                   name="critic_bn_ob")(ob_input)
        else:
            x = ob_input
        if self.config["merge_at_layer"] == 0:
            x = Concatenate(name="combined_input")([x, act_input])

        # critic hidden part
        for i, num_hiddens in enumerate(critic_hiddens):
            x = Dense(num_hiddens,
                      activation=None,
                      trainable=trainable,
                      kernel_initializer=VarianceScaling(scale=1.0 / 3, distribution="uniform"),
                      bias_initializer=VarianceScaling(scale=1.0 / 3, distribution="uniform"),
                      kernel_regularizer=l2(self.config["critic_l2"]),
                      name="critic_fc{}".format(i + 1))(x)

            # if self.config["use_bn"]:
            #     x = BatchNormalization(trainable=trainable,
            #                            name="critic_bn{}".format(i + 1))(x)

            if self.config["use_ln"]:
                x = LayerNorm(trainable=trainable, name="critic_ln{}".format(i + 1))(x)

            if self.config["use_swish"]:
                x = swish(x, name="critic_swish{}".format(i + 1))
            else:
                if lrelu > 0:
                    x = LeakyReLU(alpha=lrelu, name = "critic_lrelu{}".format(i + 1))(x)
                else:
                    x = Activation("relu", name = "critic_relu{}".format(i + 1))(x)

            if self.config["merge_at_layer"] == i + 1:
                x = Concatenate(name="combined_input")([x, act_input])

        # critic output
        qval = Dense(1, activation="linear",
                     trainable=trainable,
                     kernel_initializer=RandomUniform(minval=-3e-4, maxval=3e-4),
                     bias_initializer=RandomUniform(minval=-3e-4, maxval=3e-4),
                     kernel_regularizer=l2(self.config["critic_l2"]),
                     name="qval")(x)
        return qval

    def create_actor(self, actor_hiddens, critic_hiddens, lrelu, trainable=True):
        # actor input part
        ob_input = Input(shape=self.ob_dim, name="ob_input")
        if self.config["use_bn"]:
            x = BatchNormalization(center=False,
                                   scale=False,
                                   trainable=trainable,
                                   name="actor_bn_ob")(ob_input)
        else:
            x = ob_input

        # actor hidden part
        for i, num_hiddens in enumerate(actor_hiddens):
            x = Dense(num_hiddens,
                      activation=None,
                      trainable=trainable,
                      kernel_initializer=VarianceScaling(scale=1.0 / 3, distribution="uniform"),
                      bias_initializer=VarianceScaling(scale=1.0 / 3, distribution="uniform"),
                      kernel_regularizer=l2(self.config["actor_l2"]),
                      name="actor_fc{}".format(i + 1))(x)

            # if self.config["use_bn"]:
            #     x = BatchNormalization(trainable=trainable,
            #                            name="actor_bn{}".format(i + 1))(x)

            if self.config["use_ln"]:
                x = LayerNorm(trainable=trainable, name="actor_ln{}".format(i + 1))(x)

            if self.config["use_swish"]:
                x = swish(x, name="actor_swish{}".format(i + 1))
            else:
                if lrelu > 0:
                    x = LeakyReLU(alpha=lrelu, name = "actor_lrelu{}".format(i + 1))(x)
                else:
                    x = Activation("relu", name = "actor_relu{}".format(i + 1))(x)

        # action output
        x = Dense(self.act_dim[0],
                  activation="tanh",
                  trainable=trainable,
                  kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3),
                  bias_initializer=RandomUniform(minval=-3e-3, maxval=3e-3),
                  kernel_regularizer=l2(self.config["actor_l2"]),
                  name="action")(x)
        action = Lambda(lambda x: 0.5 * (x + 1), name="action_scaled")(x)

        # untrainable critic part
        qval = self._build_critic_part(ob_input, action, critic_hiddens, lrelu, trainable=False)

        # compile model
        actor = Model(inputs=[ob_input], outputs=[action, qval])
        actor.compile(optimizer=Adam(lr=self.config["actor_lr"]),
                      loss=[action_sqr, minus_Q],
                      loss_weights=[self.config["actor_l2_action"], 1])
        return actor

    def create_critic(self, critic_hiddens, lrelu, trainable=True):
        # critic input part
        ob_input = Input(shape=self.ob_dim, name="ob_input")
        act_input = Input(shape=self.act_dim, name="act_input")

        # critic part
        qval = self._build_critic_part(ob_input, act_input, critic_hiddens, lrelu, trainable=trainable)

        # compile
        critic = Model(inputs=[ob_input, act_input], outputs=[qval])
        optimizer = Adam(lr=self.config["critic_lr"])
        critic.compile(optimizer=optimizer, loss="mse")
        return critic

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
        actor_layers = ["action"]
        actor_layers += [l.name for l in self.actor.layers if "actor_" in l.name]
        for l in actor_layers:
            src_layer = src_model.get_layer(l)
            tar_layer = tar_model.get_layer(l)
            tau = 1.0 if "bn_ob" in l else tau
            self._copy_layer_weights(src_layer, tar_layer, tau)

    def _copy_critic_weights(self, src_model, tar_model, tau=1.0):
        critic_layers = ["qval"]
        critic_layers += [l.name for l in self.critic.layers if "critic_" in l.name]
        for l in critic_layers:
            src_layer = src_model.get_layer(l)
            tar_layer = tar_model.get_layer(l)
            tau = 1.0 if "bn_ob" in l else tau
            self._copy_layer_weights(src_layer, tar_layer, tau)

    # ==================================================== #
    #          Training Models                             #
    # ==================================================== #

    def _train_critic(self, ob0, action, reward, ob1, done):
        future_action, future_q = self.target.predict_on_batch(ob1)
        future_q = future_q.squeeze()
        reward += self.config["gamma"] * future_q * (1 - done)
        hist = self.critic.fit([ob0, action], reward,
                               batch_size=self.config["batch_size"],
                               verbose=0)
        self._copy_critic_weights(self.critic, self.actor)
        return hist

    def _train_actor(self, ob0, action, reward, ob1, done):
        # the output signals are just dummy
        hist = self.actor.fit([ob0], [reward, reward],
                              batch_size=self.config["batch_size"], verbose=0)
        return hist

    def train_actor_critic(self):
        if self.memory.size < self.config["memory_warmup"]:
            return 0, 0
        else:
            ob0, action, reward, ob1, done, steps = self.memory.sample(self.config["batch_size"])

            # mirror observation
            if self.config["mirror_ob"] and ob0 is not None:
                ob0, action, reward, ob1, done = \
                    self.ob_processor.mirror_ob(ob0, action, reward, ob1, done, steps)

            # reward shaping
            reward = self.ob_processor.reward_shaping(ob0, ob1, reward, self.config["rs_weight"])

            # reward scale
            if ob0 is not None:
                assert self.config["reward_scale"] > 0
                reward *= self.config["reward_scale"]

            # train critic
            critic_hist = self._train_critic(ob0, action, reward, ob1, done)
            # # DEBUG
            # aa, q_actor = self.actor.predict_on_batch([ob0])
            # q_critic = self.critic.predict_on_batch([ob0, aa])
            # assert np.allclose(q_actor, q_critic)
            # train actor
            actor_hist = self._train_actor(ob0, action, reward, ob1, done)
            # soft update weights
            self._copy_critic_weights(self.critic, self.target, tau=self.config["tau"])
            self._copy_actor_weights(self.actor, self.target, tau=self.config["tau"])

            return critic_hist.history["loss"][0], -1 * actor_hist.history["qval_loss"][0]

    # ==================================================== #
    #                   Trial Logic                        #
    # ==================================================== #

    def append_hist(self, hist, data):
        if hist is None:
            hist = np.copy(data)
        else:
            hist = np.vstack([hist, data])
        return hist

    def learn(self, total_episodes):
        if self.config["num_samplers"] == 0:
            return self.single_learn(total_episodes)
        else:
            return self.multi_learn(total_episodes)

    def single_learn(self, total_episodes=10000):
        episode_n = 0
        episode_reward = 0
        episode_steps = 0
        episode_losses = []
        episode_qval = []
        action_hist = None
        noisy_hist = None
        reward_hist = []
        steps_hist = []
        new_ob = self.env.reset()
        self.ob_processor.reset()

        train_step_counter = 0
        while episode_n < total_episodes:

            # select action and add noise
            new_ob = self.ob_processor.process(new_ob)
            observation = np.reshape(new_ob, [1, -1])
            action, qval = self.actor.predict(observation)
            noise = self.rand_process.sample()
            noisy_hist = self.append_hist(noisy_hist, noise)

            # apply action
            action = np.clip(action + noise, self.act_low, self.act_high)
            action_hist = self.append_hist(action_hist, action)
            act_to_apply = action.squeeze()
            if self.jump:
                act_to_apply = np.tile(act_to_apply, 2)
            new_ob, reward, done, info = self.env.step(act_to_apply)

            # bookkeeping
            episode_reward += reward
            episode_steps += 1
            train_step_counter += 1
            episode_qval.append(qval)

            # store experience
            assert np.all((action >= self.act_low) & (action <= self.act_high))
            self.memory.store(observation, action, reward, done, episode_steps)

            # train
            if train_step_counter % self.config["train_every"] == 0:
                # self.logger.info("train at episode_steps={}".format(episode_steps))
                loss, _ = self.train_actor_critic()
                if loss is not None:
                    episode_losses.append(loss)
                train_step_counter = 0

            done |= (episode_steps >= self.config["max_steps"])

            # on episode end
            if done:
                episode_n += 1
                reward_hist.append(episode_reward)
                steps_hist.append(episode_steps)

                abs_noise = np.abs(noisy_hist)
                self.logger.info(
                    "episode={0}, steps={1}, rewards={2:.4f}, avg_loss={3:.4f}, avg_q={4:.4f}, "
                    "noise=[{5:.4f}, {6:.4f}], action=[{7:.4f}, {8:.4f}]".format(episode_n,
                                                                                 episode_steps,
                                                                                 episode_reward,
                                                                                 np.mean(
                                                                                     episode_losses),
                                                                                 np.mean(
                                                                                     episode_qval),
                                                                                 np.min(abs_noise), np.max(abs_noise),
                                                                                 np.min(action_hist),
                                                                                 np.max(action_hist)
                                                                                 ))

                self.save_models()

                if episode_n % self.config["save_snapshot_every"] == 0:
                    self.save_memory()
                    self.logger.info("Snapshot saved.")

                # reset values
                episode_reward = 0
                episode_steps = 0
                episode_losses = []
                episode_qval = []
                action_hist = None
                noisy_hist = None
                new_ob = self.env.reset()
                self.ob_processor.reset()
                done = False

        self.save_models()
        self.save_memory()

        return reward_hist, steps_hist

    def multi_learn(self, total_episodes):
        import threading

        net_lock = threading.Lock()
        ob_sub_Q = Queue(maxsize=self.config["batch_size"])

        # pass a dummy observation
        self.actor.predict(np.zeros([1,55]))
        self.target.predict(np.zeros([1, 55]))
        self.critic.predict([np.zeros([1, 55]), np.zeros([1, 18])])

        def simulate(act_req_Q, act_res_Q):
            while True:
                observation = act_req_Q.get()
                net_lock.acquire()
                action, qval = self.actor.predict(observation)
                net_lock.release()
                act_res_Q.put((action, qval))

        def train():
            reward_dict = {}
            action_dict = {}
            noise_dict = {}
            qval_dict = {}

            episode_n = 0
            from collections import deque
            losses = deque(maxlen=1000)
            steps = 0
            while episode_n < total_episodes:
                msg = ob_sub_Q.get()
                pid = msg["pid"]
                observation = msg["observation"]
                done = msg["done"]
                episode_steps = msg["episode_steps"]
                action = msg["action"]
                if pid not in action_dict:
                    action_dict[pid] = None
                action_dict[pid] = self.append_hist(action_dict[pid], action)
                noise = msg["noise"]
                if pid not in noise_dict:
                    noise_dict[pid] = None
                noise_dict[pid] = self.append_hist(noise_dict[pid], noise)
                qval = msg["qval"]
                if pid not in qval_dict:
                    qval_dict[pid] = None
                qval_dict[pid] = self.append_hist(qval_dict[pid], qval)
                reward = msg["reward"]
                if pid not in reward_dict:
                    reward_dict[pid] = reward
                else:
                    reward_dict[pid] += reward
                self.memory.store(observation, action, reward, done, episode_steps)
                # self.logger.info("pid={}, noise={}".format(pid, noise))

                steps += 1
                if steps % self.config["num_samplers"] == 0:
                    net_lock.acquire()
                    loss, _ = self.train_actor_critic()
                    losses.append(loss)
                    net_lock.release()

                if done:
                    episode_n += 1

                    self.save_models()
                    if episode_n % self.config["save_snapshot_every"] == 0:
                        self.save_memory()
                        self.logger.info("Snapshot saved.")

                    abs_noise = np.abs(noise_dict[pid])
                    self.logger.info(
                        "episode={0}, steps={1}, rewards={2:.4f}, avg_loss={3:.4f}, avg_q={4:.4f}, "
                        "noise=[{5:.4f}, {6:.4f}], action=[{7:.4f}, {8:.4f}]".format(episode_n,
                                                                                     episode_steps,
                                                                                     reward_dict[pid],
                                                                                     np.mean(losses),
                                                                                     np.mean(qval_dict[pid]),
                                                                                     np.min(abs_noise),
                                                                                     np.max(abs_noise),
                                                                                     np.min(action_dict[pid]),
                                                                                     np.max(action_dict[pid])
                                                                                     ))
                    action_dict[pid] = None
                    reward_dict[pid] = 0
                    qval_dict[pid] = None
                    noise_dict[pid] = None

        th = []
        for i in xrange(self.config["num_samplers"]):
            act_req_Q = Queue(maxsize=self.config["batch_size"])
            act_res_Q = Queue(maxsize=self.config["batch_size"])
            sampler = EnvSampler(env=NIPS(max_obstacles=self.env.max_obstacles),
                                 config=self.config,
                                 act_req_Q=act_req_Q,
                                 act_res_Q=act_res_Q,
                                 ob_sub_Q=ob_sub_Q)
            sampler.start()
            simulate_thread = threading.Thread(target=simulate, args=(act_req_Q, act_res_Q))
            simulate_thread.start()
            th.append(simulate_thread)
            self.logger.info("Started one sampler thread")

        train()
        self.save_models()
        self.save_memory()
        for t in th:
            t.join()
        return None, None

    def test(self, logging=False):
        all_rewards = []
        episode_count = 0
        episode_reward = 0
        episode_steps = 0
        new_ob = self.env.reset()
        self.ob_processor.reset()

        while True:

            new_ob = self.ob_processor.process(new_ob)
            observation = np.reshape(new_ob, [1, -1])
            action, _ = self.actor.predict(observation)
            action = np.clip(action, self.act_low, self.act_high)
            act_to_apply = action.squeeze()
            if self.jump:
                act_to_apply = np.tile(act_to_apply, 2)
            new_ob, reward, done, info = self.env.step(act_to_apply)
            episode_reward += reward
            episode_steps += 1

            if logging:
                self.logger.info("episode_steps={}, episode_reward={}".format(episode_steps, episode_reward))

            done |= (episode_steps >= self.config["max_steps"])
            if done:
                episode_count += 1
                self.logger.info("Episode={}, steps={}, reward={}".format(
                    episode_count, episode_steps, episode_reward))
                all_rewards.append(episode_reward)
                episode_steps = 0
                episode_reward = 0
                new_ob = self.env.reset()
                self.ob_processor.reset()
                if not new_ob:
                    break
                if episode_count >= 10:
                    break

        return all_rewards

    def set_state(self, config):
        mem_loaded = self.load_memory()
        self.load_models(overwrite_target=(not mem_loaded))

    def save_models(self):
        paths = {"actor": "actor.h5",
                 "critic": "critic.h5",
                 "target": "target.h5"}
        paths = {k: os.path.join(self.log_dir, v) for k, v in paths.iteritems()}
        self.actor.save_weights(paths["actor"])
        self.critic.save_weights(paths["critic"])
        self.target.save_weights(paths["target"])

    def load_models(self, overwrite_target=False):
        if self.model_dir is None:
            return
        paths = {"actor": "actor.h5",
                 "critic": "critic.h5",
                 "target": "target.h5"}
        paths = {k: os.path.join(self.model_dir, v) for k, v in paths.iteritems()}
        self.logger.info("Load models from {}".format(paths))
        self.actor.load_weights(paths["actor"])
        self.critic.load_weights(paths["critic"])
        self.target.load_weights(paths["target"])
        if overwrite_target:
            self.logger.info("Overwrite target network")
            # hard copy weights
            self._copy_critic_weights(self.critic, self.actor)
            self._copy_critic_weights(self.critic, self.target)
            self._copy_actor_weights(self.actor, self.target)

    def save_memory(self):
        path = os.path.join(self.log_dir, "memory.npz")
        self.memory.save_memory(path)

    def load_memory(self):
        path = os.path.join(self.model_dir, "memory.npz")
        loaded = False
        if os.path.exists(path):
            self.memory.load_memory(path)
            self.logger.info("Loaded replay buffer from {}".format(path))
            loaded = True
        return loaded
