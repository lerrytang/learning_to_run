import numpy as np


class ReplayBuffer:
    """
    Replay buffer
    
    6 arrays for observations, actions, rewards, observation_t1, termination flags and steps
    """
    
    def __init__(self, ob_dim, act_dim, capacity=1000000):

        self._capacity = capacity

        self._observation_mem = np.zeros((capacity,) + ob_dim)
        self._action_mem = np.zeros((capacity,) + act_dim)
        self._reward_mem = np.zeros(capacity)
        self._observation_t1_mem = np.zeros((capacity,) + ob_dim)
        self._done_mem = np.ones(capacity, dtype=bool)
        self._step_mem = np.zeros(capacity)
        self.size = 0
        self._insert_index = 0
    
    def store(self, observation, action, reward, observation_t1, done, step):
        """
        Upon receiving [observation], perform [action], then receive [reward], [observation_t1] and [done]
        """
        self._observation_mem[self._insert_index] = observation
        self._action_mem[self._insert_index] = action
        self._reward_mem[self._insert_index] = reward
        self._done_mem[self._insert_index] = done
        self._observation_t1_mem[self._insert_index] = observation_t1
        self._step_mem[self._insert_index] = step
        self.size += 1
        self.size = min(self.size, self._capacity)
        self._insert_index += 1
        self._insert_index %= self._capacity
        
    def sample(self, batch_size):
        if batch_size > self.size-1:
            return None, None, None, None, None, None
        else:
            rand_ix = np.random.choice(self.size-1, batch_size, replace=False)
            obs0 = self._observation_mem[rand_ix]
            acts = self._action_mem[rand_ix]
            rewards = self._reward_mem[rand_ix]
            obs1 = self._observation_t1_mem[rand_ix]
            dones = self._done_mem[rand_ix]
            steps = self._step_mem[rand_ix]
            return obs0, acts, rewards, obs1, dones, steps

    def save_memory(self, path):
        np.savez(path,
                 capacity=self._capacity,
                 observation_mem=self._observation_mem,
                 action_mem=self._action_mem,
                 reward_mem=self._reward_mem,
                 observation_t1_mem=self._observation_t1_mem,
                 done_mem=self._done_mem,
                 step_mem=self._step_mem,
                 size=self.size,
                 insert_index=self._insert_index)

    def load_memory(self, path):
        with np.load(path) as npzfile:
            self._capacity = int(npzfile["capacity"])
            self._observation_mem = npzfile["observation_mem"]
            self._action_mem = npzfile["action_mem"]
            self._reward_mem = npzfile["reward_mem"]
            self._done_mem = npzfile["done_mem"]
            self.size = int(npzfile["size"])
            self._insert_index = int(npzfile["insert_index"])
            if "step_mem" in npzfile:
                self._step_mem = npzfile["step_mem"]
            if "observation_t1_mem" in npzfile:
                self._observation_t1_mem = npzfile["observation_t1_mem"]
            else:
                self._observation_t1_mem[:-1] = self._observation_mem[1:].copy()
                assert self._observation_t1_mem.shape == self._observation_mem.shape
                self.size -= 1
                self._insert_index -= 1
                self._insert_index %= self._capacity
