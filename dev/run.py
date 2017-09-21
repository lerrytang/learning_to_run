from osim.http.client import Client
from nipsenv import NIPS
from keras.layers import Lambda
from rand import OrnsteinUhlenbeckProcess as OUP
from mem import ReplayBuffer as RB
from agent import DDPG
from trpo import TRPO
from ob_processor import ObservationProcessor, BodySpeedAugmentor, SecondOrderAugmentor
import util

import argparse
import pickle
from datetime import datetime
import numpy as np
import sys
import os
import logging

SMTP_SERVER = None

import gym.wrappers.monitoring

# Silence the log messages
gym.envs.registration.logger.setLevel(logging.WARNING)
gym.wrappers.monitoring.logger.setLevel(logging.WARNING)


def scale_action(action):
    return Lambda(lambda x: 0.5 * (x + 1), name="action_scaled")(action)


def prepare_for_logging(name, create_folder=True):
    format_string = "%(asctime)s (pid=%(process)d) [%(levelname)s] %(message)s"
    formatter = logging.Formatter(format_string)

    current_time = datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")
    logger = logging.getLogger(name + "_" + current_time)
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


def create_rand_process(env, config):
    if "jump" in config and config["jump"]:
        act_dim = env.action_space.shape[0] / 2
    else:
        act_dim = env.action_space.shape[0]
    return OUP(
        action_dim=act_dim,
        theta=config["theta"],
        sigma_init=config["sigma_init"],
        sigma_min=config["sigma_min"],
        annealing_steps=config["annealing_steps"])


def create_memory(env, config):
    if "jump" in config and config["jump"]:
        act_dim = env.action_space.shape[0] / 2
    else:
        act_dim = env.action_space.shape[0]
    return RB(
        ob_dim=(env.observation_space.shape[0] + config["ob_aug_dim"],),
        act_dim=(act_dim,),
        capacity=config["memory_capacity"])


def train(config, trial_dir=None, visualize=False):
    pid = os.getpid()
    logger, log_dir = prepare_for_logging("pid_{}".format(pid))

    # create environment
    env = NIPS(visualize)
    logger.info("pid={}, env={}".format(pid, id(env)))
    if trial_dir is not None and os.path.exists(trial_dir):
        logger.info("Loading config from {} ...".format(trial_dir))
        with open(os.path.join(trial_dir, "config.pk"), "rb") as f:
            config = pickle.load(f)
    config["scale_action"] = scale_action
    config["title_prefix"] = "RunEnv"

    # observation processor
    if "ob_processor" not in config or config["ob_processor"] == "dummy":
        ob_processor = ObservationProcessor()
    elif config["ob_processor"] == "2ndorder":
        ob_processor = SecondOrderAugmentor()
    else:
        ob_processor = BodySpeedAugmentor()
    config["ob_aug_dim"] = ob_processor.get_aug_dim()

    # snapshot info
    if "save_snapshot_every" not in config:
        config["save_snapshot_every"] = 500
    save_snapshot_every = config["save_snapshot_every"]

    # save config
    with open(os.path.join(log_dir, "config.pk"), "wb") as f:
        pickle.dump(config, f)
    util.print_settings(logger, config, env)

    # DDPG
    if config['agent'] == 'DDPG':
        # create random process
        oup = create_rand_process(env, config)

        # create replay buffer
        memory = create_memory(env, config)

        # create ddpg agent
        agent = DDPG(env, memory, oup, ob_processor, config)
        agent.build_nets(
            actor_hiddens=config["actor_hiddens"],
            scale_action=config["scale_action"],
            critic_hiddens=config["critic_hiddens"])

        # print networks
        agent.actor.summary()
        agent.target_actor.summary()
        agent.critic.summary()

        # add callbacks
        def p_info(episode_info):
            util.print_episode_info(logger, episode_info, pid)

        def save_nets(episode_info):
            paths = {}
            paths["actor"] = os.path.join(log_dir, "actor.h5")
            paths["critic"] = os.path.join(log_dir, "critic.h5")
            paths["target"] = os.path.join(log_dir, "target.h5")
            agent = episode_info["agent"]
            agent.save_models(paths)

        def save_snapshots(episode_info):
            agent = episode_info["agent"]
            episode = episode_info["episode"]
            if episode % save_snapshot_every == 0:
                paths = {}
                paths["actor"] = os.path.join(log_dir, "actor_{}.h5".format(episode))
                paths["critic"] = os.path.join(log_dir, "critic_{}.h5".format(episode))
                paths["target"] = os.path.join(log_dir, "target_{}.h5".format(episode))
                agent.save_models(paths)
                memory_path = os.path.join(log_dir, "replaybuffer.npz")
                agent.save_memory(memory_path)
                logger.info("Snapshots saved. (pid={})".format(pid))

        agent.on_episode_end.append(p_info)
        agent.on_episode_end.append(save_nets)
        agent.on_episode_end.append(save_snapshots)

        # load existing model
        if trial_dir is not None and os.path.exists(trial_dir):
            logger.info("Loading networks from {} ...".format(trial_dir))
            paths = {}
            paths["actor"] = "actor.h5"
            paths["critic"] = "critic.h5"
            paths["target"] = "target.h5"
            paths = {k: os.path.join(trial_dir, v) for k, v in paths.iteritems()}
            logger.info("Paths to models: {}".format(paths))
            agent.load_models(paths)
            memory_path = os.path.join(trial_dir, "replaybuffer.npz")
            if os.path.exists(memory_path):
                agent.load_memory(memory_path)
                logger.info("Replay buffer loaded.")

        # learn
        util.print_sec_header(logger, "Training")
        reward_hist, steps_hist = agent.learn(
            total_episodes=config["total_episodes"],
            max_steps=config["max_steps"])
        env.close()

    # TRPO
    elif config['agent'] == 'TRPO':

        def env_maker():
            env = NIPS(visualize=False)
            monitor_dir = os.path.join(log_dir, "gym_monitor")
            env = gym.wrappers.Monitor(env, directory=monitor_dir, video_callable=False, force=False, resume=True,
                                       write_upon_reset=True)
            return env

        del env
        env = env_maker()

        agent = TRPO(env,
                     env_maker,
                     logger,
                     n_envs=4,
                     batch_size=5000,
                     n_iters=5000
                     )
        agent.learn()

    # send result
    img_file = os.path.join(log_dir, "train_stats.png")
    util.plot_stats(reward_hist, steps_hist, img_file)
    log_file = os.path.join(log_dir, "train.log")
    title = log_dir + "_" + config["title_prefix"]
    util.send_email(title, [img_file], [log_file], SMTP_SERVER)

    logger.info("Finished (pid={}).".format(pid))


def test(trial_dir, test_episode, visual_flag, submit_flag):
    pid = os.getpid()
    logger, _ = prepare_for_logging("pid_{}".format(pid), False)

    logger.info("trial_dir={}".format(trial_dir))
    if not os.path.exists(trial_dir):
        logger.info("trial_dir does not exist")
        return

    # create environment
    env = NIPS(visualize=visual_flag)

    # load config
    with open(os.path.join(trial_dir, "config.pk"), "rb") as f:
        config = pickle.load(f)
    config["scale_action"] = scale_action

    # observation processor
    if "ob_processor" not in config or config["ob_processor"] == "dummy":
        ob_processor = ObservationProcessor()
    elif config["ob_processor"] == "2ndorder":
        ob_processor = SecondOrderAugmentor()
    else:
        ob_processor = BodySpeedAugmentor()
    config["ob_aug_dim"] = ob_processor.get_aug_dim()
    util.print_settings(logger, config, env)

    # create random process
    oup = create_rand_process(env, config)

    # create replay buffer
    memory = create_memory(env, config)

    # create ddpg agent
    agent = DDPG(env, memory, oup, ob_processor, config)
    agent.build_nets(
        actor_hiddens=config["actor_hiddens"],
        scale_action=config["scale_action"],
        critic_hiddens=config["critic_hiddens"])

    # load weights
    paths = {}
    if test_episode > 0:
        paths["actor"] = "actor_{}.h5".format(test_episode)
        paths["critic"] = "critic_{}.h5".format(test_episode)
        paths["target"] = "target_{}.h5".format(test_episode)
    else:
        paths["actor"] = "actor.h5"
        paths["critic"] = "critic.h5"
        paths["target"] = "target.h5"
    paths = {k: os.path.join(trial_dir, v) for k, v in paths.iteritems()}
    logger.info("Paths to models: {}".format(paths))
    agent.load_models(paths)

    if submit_flag:
        submit(agent, logger)
    else:
        rewards = []
        for i in xrange(10):
            steps, reward = agent.test(max_steps=1000)
            logger.info("episode={}, steps={}, reward={}".format(i, steps, reward))
            rewards.append(reward)
        logger.info("avg_reward={}".format(np.mean(rewards)))


def submit(agent, logger, jump=False):
    token = None
    assert token is not None, "You need to provide your token to submit()"
    # Settings
    remote_base = 'http://grader.crowdai.org:1729'
    client = Client(remote_base)
    # Create environment
    new_ob = client.env_create(token)
    agent.ob_processor.reset()
    zero_action = np.zeros(agent.env.action_space.shape).tolist()
    first_frame = True
    done = False
    # Run a single step
    # The grader runs 3 simulations of at most 1000 steps each. We stop after the last one
    episode_count = 0
    episode_steps = 0
    episode_reward = 0

    all_rewards = []

    while True:

        # ignore first frame because it contains phantom obstacle
        if first_frame:
            new_ob, reward, done, info = client.env_step(zero_action, True)
            episode_reward += reward
            episode_steps += 1
            first_frame = False
            assert not done, "Episode finished in one step"
            continue

        new_ob = agent.ob_processor.process(new_ob)
        observation = np.reshape(new_ob, [1, -1])
        action, _ = agent.actor.predict(observation)
        action = np.clip(action, agent.act_low, agent.act_high)
        act_to_apply = action.squeeze()
        if self.jump:
            act_to_apply = np.tile(act_to_apply, 2)
        [new_ob, reward, done, info] = client.env_step(act_to_apply.tolist(), True)

        episode_steps += 1
        episode_reward += reward
        logger.info("step={}, reward={}".format(episode_steps, reward))

        if done:
            episode_count += 1
            logger.info("Episode={}, steps={}, reward={}".format(
                episode_count, episode_steps, episode_reward))
            all_rewards.append(episode_reward)

            episode_steps = 0
            episode_reward = 0
            new_ob = client.env_reset()
            agent.ob_processor.reset()
            first_frame = True
            if not new_ob:
                break
    client.submit()
    logger.info("All rewards: {}".format(all_rewards))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test DDPG")
    parser.add_argument('--train', dest='train', action='store_true', default=True)
    parser.add_argument('--test', dest='train', action='store_false', default=True)
    parser.add_argument('--submit', dest='submit', action='store_true', default=False)
    parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
    parser.add_argument('--trial_dir', dest='trial_dir', default=None, type=str)
    parser.add_argument('--test_episode', dest='test_episode', default=0, type=int)
    args = parser.parse_args()

    if args.train:
        config = {
            "use_bn": True,
            "save_snapshot_every": 500,
            "num_train": 2,
            "jump": False,
            "gamma": 0.99,
            "tau": 1e-3,
            "batch_size": 128,
            "actor_l2": 1e-6,
            "actor_lr": 1e-4,
            "actor_l2_action": 1e-5,
            "critic_l2": 1e-6,
            "critic_lr": 3.25e-4,
            "merge_at_layer": 1,
            "theta": 0.15,
            "sigma_init": 0.1,
            "sigma_min": 0.002,
            "total_episodes": 50000,
            "max_steps": 1000,
            "memory_warmup": 1000,
            "memory_capacity": 1000000,
            "annealing_steps": 3000000,
            "actor_hiddens": [128, 128, 64, 64],
            "critic_hiddens": [128, 128, 64, 64],
            "scale_action": scale_action,
            "title_prefix": "RunEnv",
            "ob_processor": "bodyspeed",  # 1st order system
            #                "ob_processor": "2ndorder",
        }

        train(config, args.trial_dir, args.visualize)
    else:
        test(args.trial_dir, args.test_episode, args.visualize, args.submit)
