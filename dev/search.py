import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from run import train
from util import load_config
from copy import deepcopy
from multiprocessing import Pool


if __name__ == "__main__":

    pid = os.getpid()
    print "pid={}".format(pid)

    # parameters space to search
    actor_l2_actions = [1e-6, 1e-5, 1e-4, 1e-3]
    clear_vels = [True, False]
    include_limb_vels = [True, False]
    net_archs = [[128, 128, 64, 64],
                 [512, 256, 128]]

    config = load_config("default.yaml")["DDPG"]
    config["agent"] = "DDPG"

    # workers
    MAX_NUM_WORKERS = 8
    worker_pool = Pool(processes=MAX_NUM_WORKERS)

    # grid search
    for actor_l2_action in actor_l2_actions:
        for clear_vel in clear_vels:
            for include_limb_vel in include_limb_vels:
                for net_arch in net_archs:
                    p_config = deepcopy(config)
                    p_config["actor_l2_action"] = actor_l2_action
                    p_config["clear_vel"] = clear_vel
                    p_config["include_limb_vel"] = include_limb_vel
                    p_config["actor_hiddens"] = net_arch
                    p_config["critic_hiddens"] = net_arch
                    worker_pool.apply_async(func=train, args=(p_config,))

    # clean up
    worker_pool.close()
    worker_pool.join()

