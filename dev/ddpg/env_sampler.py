from multiprocessing import Process
import util
import numpy as np
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class EnvSampler(Process):

    def __init__(self, env, config, act_req_Q, act_res_Q, ob_sub_Q):
        super(EnvSampler, self).__init__()
        self.env = env
        self.ob_processor = util.create_ob_processor(env, config)
        self.rand_process = util.create_rand_process(env, config)
        self.max_steps = config["max_steps"]
        self.total_episodes = config["total_episodes"]
        self.act_req_Q = act_req_Q
        self.act_res_Q = act_res_Q
        self.ob_sub_Q = ob_sub_Q
        real_act_dim = self.env.action_space.shape[0]
        self.act_low = self.env.action_space.high[:real_act_dim]
        self.act_high = self.env.action_space.low[:real_act_dim]

    def run(self):
        logger.info("EnvSampler started, pid={}".format(self.pid))
        episode_n = 0
        episode_steps = 0
        episode_reward = 0
        new_ob = self.env.reset()
        self.ob_processor.reset()
        while episode_n < self.total_episodes:

            try:
                # request for action and add noise
                new_ob = self.ob_processor.process(new_ob)
                observation = np.reshape(new_ob, [1, -1])
                self.act_req_Q.put(observation)
                action, qval = self.act_res_Q.get()
            except:
                break

            noise = self.rand_process.sample()
            # logger.info("pid={}, noise={}".format(self.pid, noise))

            # apply action
            action = np.clip(action + noise, self.act_low, self.act_high)
            act_to_apply = action.squeeze()
            new_ob, reward, done, info = self.env.step(act_to_apply)

            # bookkeeping
            episode_steps += 1
            episode_reward += reward

            # send experience back to agent
            msg = {"pid": self.pid,
                   "observation": observation,
                   "action": action,
                   "reward": reward,
                   "done": done,
                   "episode_steps": episode_steps,
                   "qval": qval,
                   "noise": noise}
            self.ob_sub_Q.put(msg)

            done |= (episode_steps >= self.max_steps)

            # on episode end
            if done:
                episode_n += 1

                logger.info(
                    "pid={0}, episode={1}, steps={2}, rewards={3:.4f}".format(self.pid,
                                                                              episode_n,
                                                                              episode_steps,
                                                                              episode_reward))

                # reset values
                episode_reward = 0
                episode_steps = 0
                new_ob = self.env.reset()
                self.ob_processor.reset()
        self.env.close()