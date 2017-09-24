from osim.env import RunEnv
from osim.http.client import Client


GRADER_URL = 'http://grader.crowdai.org:1729'


class NIPS(object):

    def __init__(self, visualize=False, token=None):
        if token is None:
            self.remote_env = False
            self.env = RunEnv(visualize=visualize)
        else:
            self.remote_env = True
            self.token = token
            self.env = Client(GRADER_URL)

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self):
        if self.remote_env:
            ob = self.env.env_create(self.token)
        else:
            ob = self.env.reset()
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

