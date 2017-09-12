import os
import numpy as np
from proportional import Memory
import pickle
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ReplayBuffer(object):
    """
    Replay buffer
    
    4 arrays for observations, actions, rewards and termination flags
    """
    
    def __init__(self, ob_dim, act_dim, capacity=1000000):

        self._capacity = capacity

        self._observation_mem = np.zeros((capacity,) + ob_dim)
        self._action_mem = np.zeros((capacity,) + act_dim)
        self._reward_mem = np.zeros(capacity)
        self._done_mem = np.ones(capacity, dtype=bool)
        self.size = 0
        self._insert_index = 0
    
    def store(self, observation, action, reward, done):
        """
        Upon receiving [observation], perform [action], then receive [reward] and [done]
        """
        self._observation_mem[self._insert_index] = observation
        self._action_mem[self._insert_index] = action
        self._reward_mem[self._insert_index] = reward
        self._done_mem[self._insert_index] = done
        self.size += 1
        self.size = min(self.size, self._capacity)
        self._insert_index += 1
        self._insert_index = self._insert_index % self._capacity
        
    def sample(self, batch_size):
        if batch_size > self.size-1:
            return (None, None, None, None, None), None, None
        else:
            rand_ix = np.random.choice(self.size-1, batch_size, replace=False)
            obs0 = self._observation_mem[rand_ix]
            acts = self._action_mem[rand_ix]
            rewards = self._reward_mem[rand_ix]
            dones = self._done_mem[rand_ix]
            ix = rand_ix + 1
            ix = ix % self.size
            obs1 = self._observation_mem[ix]
            return (obs0, acts, rewards, obs1, dones), None, None

    def update_priority(self, indices, priorities):
        pass

    def save_memory(self, path):
        np.savez(path,
                capacity=self._capacity,
                observation_mem=self._observation_mem,
                action_mem=self._action_mem,
                reward_mem=self._reward_mem,
                done_mem=self._done_mem,
                size=self.size,
                insert_index=self._insert_index)

    def load_memory(self, path):
        npzfile = np.load(path)
        self._capacity = npzfile["capacity"]
        self._observation_mem = npzfile["observation_mem"]
        self._action_mem = npzfile["action_mem"]
        self._reward_mem = npzfile["reward_mem"]
        self._done_mem = npzfile["done_mem"]
        self.size = npzfile["size"]
        self._insert_index = npzfile["insert_index"]
        

# ========================================================================


class PrioritizedReplayBuffer(ReplayBuffer):

    def __init__(self, ob_dim, act_dim, capacity=1000000, epsilon=1e-8,
            alpha=0.7, beta=0.4, beta_anneal_rate=1e-5, abs_err_upper=1.0):
        super(PrioritizedReplayBuffer, self).__init__(ob_dim, act_dim, capacity)
        self.ix_mem = Memory(capacity, epsilon, alpha, beta, beta_anneal_rate, abs_err_upper)

    def store(self, observation, action, reward, done):
        super(PrioritizedReplayBuffer, self).store(observation, action, reward, done)
        ix = (self._insert_index - 1) % self._capacity
        self.ix_mem.store(ix)

    def sample(self, batch_size):
        if batch_size > self.size-1:
            return (None, None, None, None, None), None, None
        indices, data_ix, weights = self.ix_mem.sample(batch_size)
        data_ix = np.asarray(data_ix)
        obs0 = self._observation_mem[data_ix]
        acts = self._action_mem[data_ix]
        rewards = self._reward_mem[data_ix]
        dones = self._done_mem[data_ix]
        ix = data_ix + 1
        ix = ix % self.size
        obs1 = self._observation_mem[ix]
#        logger.info("data_ix={}, self.size={}, self._insert_index={}".format(
#            data_ix, self.size, self._insert_index))
#        logger.info("indices={}".format(indices))
        return (obs0, acts, rewards, obs1, dones), weights, indices

    def update_priority(self, indices, abs_td_error):
        self.ix_mem.batch_update(indices, abs_td_error)
    
    def save_memory(self, path):
        super(PrioritizedReplayBuffer, self).save_memory(path)
        basename = os.path.dirname(path)
        experience_path = os.path.join(basename, "ix_mem.pkl")
        with open(experience_path, "wb") as f:
            pickle.dump(self.ix_mem, f)

    def load_memory(self, path):
        super(PrioritizedReplayBuffer, self).load_memory(path)
        basename = os.path.dirname(path)
        experience_path = os.path.join(basename, "ix_mem.pkl")
        with open(experience_path, "rb") as f:
            self.ix_mem = pickle.load(f)


