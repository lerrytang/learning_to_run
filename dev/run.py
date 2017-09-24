import argparse
from nipsenv import NIPS
import util
from datetime import datetime
import sys
import os
import logging
import numpy as np

SUPPORTED_AGENTS = ["DDPG", "TRPO"]


def prepare_for_logging(name, create_folder=True):
    format_string = "%(asctime)s (pid=%(process)d) [%(levelname)s] %(message)s"
    formatter = logging.Formatter(format_string)

    current_time = datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")
    logger = logging.getLogger(current_time + "_" + name)
    logger.setLevel(logging.INFO)

    log_dir = None
    if create_folder:
        # create folder to save log and results
        dirname = name + "_" + current_time
        log_dir = os.path.join("trials", dirname)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # set up logger
        log_file = os.path.join(log_dir, "train.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger, log_dir


def train(config, trial_dir=None, visualize=False):
    t_agent = config["agent"]
    assert t_agent in SUPPORTED_AGENTS, "Agent type {} not supported".format(t_agent)

    # prepare trial environment
    pid = os.getpid()
    trial_name = "{}_pid{}".format(t_agent, pid)
    logger, log_dir = prepare_for_logging(trial_name)

    # create agent
    env = NIPS(visualize)
    logger.info("pid={}, env={}".format(pid, id(env)))

    # if old config is found, new config entris overwrites those in the old
    fine_tuning = False
    if trial_dir is not None:
        config_file = os.path.join(trial_dir, "config.yaml")
        if os.path.exists(config_file):
            logger.info("Found config file in {}".format(trial_dir))
            existing_config = util.load_config(config_file)
            assert existing_config["agent"] == t_agent, "Different algorithms"
            fine_tuning = True
            for k, v in config.iteritems():
                existing_config[k] = v
            config = existing_config
            config["model_dir"] = trial_dir

    # save config to the trial folder
    util.print_settings(logger, config, env)
    config_file = os.path.join(log_dir, "config.yaml")
    util.save_config(config_file, config)

    # instantiate an agent
    config["logger"] = logger
    config["log_dir"] = log_dir
    if t_agent == "DDPG":
        from ddpg import DDPG
        agent = DDPG(env, config)
    elif t_agent == "TRPO":
        from trpo import TRPO
        agent = TRPO(env, config)
    else:
        # because of the assertion above, this should never happen
        raise ValueError("Unsupported agent type: {}".format(t_agent))

    # learn
    if fine_tuning:
        util.print_sec_header(logger, "Continual training")
        agent.set_state(config)
    else:
        util.print_sec_header(logger, "Training from scratch")
    reward_hist, steps_hist = agent.learn(total_episodes=config["total_episodes"])
    env.close()

    # send result
    img_file = os.path.join(log_dir, "train_stats.png")
    util.plot_stats(reward_hist, steps_hist, img_file)
    log_file = os.path.join(log_dir, "train.log")
    util.send_email(log_dir, [img_file], [log_file], config)

    logger.info("Finished (pid={}).".format(pid))


def test(trial_dir, visual_flag, token):
    assert trial_dir is not None and os.path.exists(trial_dir)

    env = NIPS(visual_flag, token)

    # prepare trial environment
    pid = os.getpid()
    logger, _ = prepare_for_logging(str(pid), create_folder=False)

    # load config
    config_file = os.path.join(trial_dir, "config.yaml")
    config = util.load_config(config_file)
    util.print_settings(logger, config, env)

    # instantiate an agent
    config["logger"] = logger
    config["log_dir"] = trial_dir
    t_agent = config["agent"]
    if t_agent == "DDPG":
        from ddpg import DDPG
        agent = DDPG(env, config)
    elif t_agent == "TRPO":
        from trpo import TRPO
        agent = TRPO(env, config)
    else:
        raise ValueError("Unsupported agent type: {}".format(t_agent))
    agent.set_state(config)

    # test
    util.print_sec_header(logger, "Testing")
    rewards = agent.test()
    logger.info("avg_reward={}".format(np.mean(rewards)))
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', default='TRPO', choices=SUPPORTED_AGENTS)
    parser.add_argument('--config_yaml', default=None, type=str)
    parser.add_argument('--test', dest='train', action='store_false', default=True)
    parser.add_argument('--token', default=None, type=str)
    parser.add_argument('--visualize', action='store_true', default=False)
    parser.add_argument('--trial_dir', default=None, type=str)
    args = parser.parse_args()

    if args.train:
        if args.config_yaml is None or not os.path.exists(args.config_yaml):
            config = util.load_config("default.yaml")
            config = config[args.agent]
        else:
            config = util.load_config(args.config_yaml)
        config["agent"] = args.agent
        train(config, args.trial_dir, args.visualize)
    else:
        test(args.trial_dir, args.visualize, args.token)
