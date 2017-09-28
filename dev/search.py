import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
from run import train
from util import load_config
from copy import deepcopy
from multiprocessing import Pool


"""
use_bn: False
save_snapshot_every: 1000
train_every: 4
jump: False
gamma: 0.99
tau: 0.001
batch_size: 128
actor_l2: 0.000001
actor_lr: 0.0001
actor_l2_action: 0.000001
critic_l2: 0.000001
critic_lr: 0.0003
merge_at_layer: 1
theta: 0.15
sigma_init: 0.2
sigma_min: 0.002
total_episodes: 30000
max_steps: 1000
memory_warmup: 10000
memory_capacity: 1000000
annealing_steps: 4000000
actor_hiddens: [128, 128, 64, 64]
critic_hiddens: [128, 128, 64, 64]
ob_processor: "bodyspeed"
lrelu: -1
"""

if __name__ == "__main__":

    pid = os.getpid()
    print "pid={}".format(pid)

    # parameters space to search
    ob_processors = ["norm1storder", "bodyspeed"]
    mirror_obs = [True, False]
    toe_dist_thresholds = [0.1, 0.2]
    annealing_steps = [2000000, 4000000, 8000000, 16000000]
    train_every_s = np.arange(5) + 1
    actor_l2_actions = np.logspace(-6, -3, 5)
    jumps = [False, True]
    net_archs = [[128, 128, 64, 64],
            [256, 256, 128, 64],
            [512, 512, 128],
            [1024, 512]]

    config = load_config("default.yaml")["DDPG"]
    config["agent"] = "DDPG"

    # workers
    MAX_NUM_WORKERS = 8
    worker_pool = Pool(processes=MAX_NUM_WORKERS)

    # grid search
    for train_every in train_every_s:
        for ob_processor in ob_processors:
            for mirror_ob in mirror_obs:
                for toe_dist_threshold in toe_dist_thresholds:
                    for annealing_step in annealing_steps:
                        for actor_l2_action in actor_l2_actions:
                            for jump in jumps:
                                for net_arch in net_archs:
                                    p_config = deepcopy(config)
                                    p_config["train_every"] = train_every
                                    p_config["ob_processor"] = ob_processor
                                    p_config["mirror_ob"] = mirror_ob
                                    p_config["toe_dist_threshold"] = toe_dist_threshold
                                    p_config["annealing_steps"] = annealing_step
                                    p_config["actor_l2_action"] = actor_l2_action
                                    p_config["jump"] = jump
                                    p_config["actor_hiddens"] = net_arch
                                    p_config["critic_hiddens"] = net_arch
                                    worker_pool.apply_async(func=train, args=(p_config,))

    # clean up
    worker_pool.close()
    worker_pool.join()

