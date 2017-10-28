import numpy as np


class OrnsteinUhlenbeckProcess:
    """
    Ornstein Uhlenbeck Process
    """
    
    def __init__(self, action_dim, theta, sigma_init, init_val=0.0, sigma_min=0, annealing_steps=0):
        self.action_dim = action_dim
        self.theta = theta

        self.noise_var = np.ones(action_dim) * init_val
        self.sigma = sigma_init
        self.sigma_min = max(sigma_min, 0)
        assert self.sigma>=self.sigma_min
        
        if annealing_steps>0:
            self.sigma_delta = (self.sigma - self.sigma_min) / annealing_steps
        else:
            self.sigma_delta = 0
        
    def sample(self):
        noise = self.noise_var * self.theta - np.random.randn(self.action_dim) * self.sigma
        self.noise_var -= noise
        self.sigma -= self.sigma_delta
        self.sigma = max(self.sigma, self.sigma_min)
        return noise


class OUPfromWiki:
    """
    Ornstein Uhlenbeck Process whose implementation follows wikipedia
    """

    def __init__(self, action_dim, theta, sigma, init_val=0.0, scale_min=0, annealing_steps=0, seed=0):
        self.action_dim = action_dim
        self.theta = theta
        self.sigma = sigma

        self.scale = 1.0
        self.scale_min = scale_min
        if annealing_steps > 0:
            self.scale_delta = (self.scale - scale_min) / annealing_steps
        else:
            self.scale_delta = 0.0

        # initialize x0
        self.xt = np.ones(action_dim) * init_val

        np.random.seed(seed)

    def sample(self):
        """
        {\displaystyle dx_{t}=\theta (\mu -x_{t})\,dt+\sigma \,dW_{t}}
        :return:
        """
        delta_xt = self.theta * (-1.0 * self.xt) + self.sigma * np.random.randn(self.action_dim)
        self.xt += delta_xt
        noise = self.scale * self.xt
        self.scale = max(self.scale - self.scale_delta, self.scale_min)
        return noise


