"""
All RL algorithms should implement the following methods:
1. learn()             - learn the agent
2. test(token)         - test agent's performance
3. set_state(config)   - set agent's status for continual training
"""


class Agent(object):

    def __init__(self, env, config):
        pass

    def learn(self):
        """
        All necessary settings are in config
        """
        raise NotImplementedError()

    def test(self, token):
        """
        If token is not known, submit to the grading server.
        """
        raise NotImplementedError()

    def set_state(self, config):
        """
        Perform operations for continual training.
        E.g., load networks' weights
        """
        raise NotImplementedError()
