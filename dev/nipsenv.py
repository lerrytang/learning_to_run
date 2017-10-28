from osim.env import RunEnv
from osim.http.client import Client
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

GRADER_URL = 'http://grader.crowdai.org:1729'


class NIPS(object):

    def __init__(self, visualize=False, token=None, max_obstacles=3):
        logger.info("max_obstacles={}".format(max_obstacles))
        self.max_obstacles = max_obstacles
        if token is None:
            self.remote_env = False
            self.env = RunEnv(visualize=visualize, max_obstacles=max_obstacles)
        else:
            self.remote_env = True
            self.local_env = RunEnv(visualize=False, max_obstacles=max_obstacles)
            self.token = token
            self.env = Client(GRADER_URL)
            self.env_created = False

    @property
    def observation_space(self):
        if self.remote_env:
            # because Client() has not observation_space
            return self.local_env.observation_space
        else:
            return self.env.observation_space

    @property
    def action_space(self):
        if self.remote_env:
            # because Client() has not action_space
            return self.local_env.action_space
        else:
            return self.env.action_space

    def reset(self):
        if self.remote_env:
            if not self.env_created:
                ob = self.env.env_create(self.token)
                self.env_created = True
            else:
                ob = self.env.env_reset()
        else:
            ob = self.env.reset(difficulty=2)
        return ob

    def step(self, action):
        if self.remote_env:
            ob, reward, done, info = self.env.env_step(action.tolist(), True)
        else:
            ob, reward, done, info = self.env.step(action)
        return ob, reward, done, info

    def close(self):
        if self.remote_env:
            self.env.submit()
        else:
            self.env.close()

