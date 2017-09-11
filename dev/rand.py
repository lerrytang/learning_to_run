import numpy as np


class OrnsteinUhlenbeckProcess:
    """
    Ornstein Uhlenbeck Process
    """
    
    def __init__(self, action_dim, theta, sigma_init, sigma_min=0, annealing_steps=0):
        self.action_dim = action_dim
        self.theta = theta

        self.noise_var = np.zeros(action_dim)
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


if __name__ == "__main__":

    import logging
    logging.basicConfig(level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    num_rand = 100
    action_dim = 3
    sigma_init = 0.2
    sigma_min = 0
    annealing_steps = 100

    oup = OrnsteinUhlenbeckProcess(
            action_dim=action_dim,
            theta=.15,
            sigma_init=sigma_init,
            sigma_min=sigma_min,
            annealing_steps=annealing_steps)

    samples = np.zeros([num_rand, action_dim])
    for i in xrange(num_rand):
        samples[i] = oup.sample()

    logger.info("sigma_init={}, sigma_min={}, annealing_steps={}".format(
        sigma_init, sigma_min, annealing_steps))
    logger.info("mean(samples)={}".format(samples.mean(axis=0)))
    logger.info("std(samples)={}".format(samples.std(axis=0)))

