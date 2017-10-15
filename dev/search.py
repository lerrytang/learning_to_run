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
    lrelus = [-1, 0.1, 0.3, 0.5]
    actor_l2_actions = [1e-6, 1e-5, 1e-4, 1e-3]
    use_lns = [True, False]

    config = load_config("default.yaml")["DDPG"]
    config["agent"] = "DDPG"

    # workers
    MAX_NUM_WORKERS = 8
    worker_pool = Pool(processes=MAX_NUM_WORKERS)

    # grid search
    for actor_l2_action in actor_l2_actions:
        for lrelu in lrelus:
            for use_ln in use_lns:
                p_config = deepcopy(config)
                p_config["actor_l2_action"] = actor_l2_action
                p_config["lrelu"] = lrelu
                p_config["use_ln"] = use_ln
                worker_pool.apply_async(func=train, args=(p_config,))

    # clean up
    worker_pool.close()
    worker_pool.join()

