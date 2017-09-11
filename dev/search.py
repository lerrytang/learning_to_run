import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
from run import train
from copy import deepcopy
from multiprocessing import Pool


config = {
        "gamma": 0.99,
        "tau": 1e-3,
        "batch_size": 128,
        "actor_l2": 1e-6,
        "actor_lr": 1e-4,
        "critic_l2": 1e-6,
        "critic_lr": 1e-3,
        "merge_at_layer": 1,
        "theta": 0.15,
        "sigma_init": 0.2,
        "sigma_min": 0.05,
        "total_episodes": 10000,
        "max_steps": 1000,
        "memory_capacity": 1000000,
        "annealing_steps": 2000000,
        "actor_hiddens": [128, 128, 64, 64],
        "critic_hiddens": [128, 128, 64, 64],
        "scale_action": None,
        "title_prefix": "RunEnv",
        "ob_processor": "bodyspeed"
        }


if __name__ == "__main__":

    # parameters space to search
    gammas = [0.99, 0.995]
    taus = np.logspace(-3, -2, 10)
    critic_lrs = np.linspace(1e-4, 1e-3, 5)
    sigma_inits = [0.2, 0.25, 0.3, 0.35]

    # workers
    MAX_NUM_WORKERS = 7
    worker_pool = Pool(processes=MAX_NUM_WORKERS)

    # grid search
    for gamma in gammas:
        for tau in taus:
            for critic_lr in critic_lrs:
                for sigma_init in sigma_inits:
                    p_config = deepcopy(config)
                    p_config["gamma"] = gamma
                    p_config["tau"] = tau
                    p_config["critic_lr"] = critic_lr
                    p_config["sigma_init"] = sigma_init
                    worker_pool.apply_async(func=train, args=(p_config,))

    # clean up
    worker_pool.close()
    worker_pool.join()

