import yaml
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import numpy as np
import socket
import pickle

from ddpg.rand import OUPfromWiki as OUP
from ob_processor import ObservationProcessor, BodySpeedAugmentor, SecondOrderAugmentor
from ob_processor import NormalizedFirstOrder, SecondRound
import random
import numpy as np

# ============================================= #
#             DDPG related                      #
# ============================================= #

def process_ob(ob_processor, ob):
    ob = ob_processor.process(ob)
    return np.reshape(ob, [1, -1])


def create_rand_process(env, config):
    if "jump" in config and config["jump"]:
        act_dim = env.action_space.shape[0] / 2
    else:
        act_dim = env.action_space.shape[0]
    return OUP(
        action_dim=act_dim,
        theta=config["theta"],
        sigma=config["sigma_init"],
        sigma_min=config["sigma_min"],
        annealing_steps=config["annealing_steps"],
        seed=random.randint(0, 10000))


def create_ob_processor(env, config):
    if "ob_processor" not in config or config["ob_processor"] == "dummy":
        obp = ObservationProcessor()
    elif config["ob_processor"] == "2ndorder":
        obp = SecondOrderAugmentor()
    elif config["ob_processor"] == "norm1storder":
        obp = NormalizedFirstOrder()
    elif config["ob_processor"] == "2ndround":
        obp = SecondRound(max_num_ob=config["max_obstacles"],
                          ob_dist_scale=config["ob_dist_scale"],
                          fake_ob_pos=config["fake_ob_pos"],
                          clear_vel=config["clear_vel"],
                          include_limb_vel=config["include_limb_vel"])
    else:
        obp = BodySpeedAugmentor(max_num_ob=config["max_obstacles"],
                          fake_ob_pos=config["fake_ob_pos"])
    return obp

# ============================================= #
#             Pretty Printing                   #
# ============================================= #

def print_sec_header(logger, header):
    logger.info("-"*30)
    logger.info("< {} >".format(header))
    logger.info("-"*30)


def print_settings(logger, config, env=None):
    print_sec_header(logger, "Parameters")
    for k, v in config.iteritems():
        logger.info("{}: {}".format(k, v))
    logger.info("")
    if env is not None:
        print_sec_header(logger, "Environment Spec")
        logger.info("env.observation_space={}".format(env.observation_space))
        logger.info("env.action_space={}".format(env.action_space))
        logger.info("env.action_space.high={}".format(env.action_space.high))
        logger.info("env.action_space.low={}".format(env.action_space.low))
        logger.info("")


# ============================================= #
#          Configration Utilities               #
# ============================================= #

def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f)
    return config


def save_config(config_file, config):
    with open(config_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def load_legacy_config(config_pk):
    with open(config_pk, "rb") as f:
        config = pickle.load(f)
    return config


# ============================================= #
#             Reporting Utilities               #
# ============================================= #

def plot_stats(reward_hist, steps_hist, img_file):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    ix = np.arange(len(reward_hist)) + 1
    pd.Series(reward_hist, index=ix).plot(ax=axes[0], color="blue")
    axes[0].set_title("Rewards Per Episode")
    pd.Series(steps_hist, index=ix).plot(ax=axes[1], color="red")
    axes[1].set_title("Steps Per Episode")
    plt.savefig(img_file)


def send_email(title, image_files, txt_files, config):

    smtp_server = None if "smtp" not in config else config["smtp"]
    if smtp_server is None:
        return

    me = config["mail_from"]
    you = config["mail_to"]
    hostname = socket.gethostname()
    msg = MIMEMultipart()
    msg['Subject'] = hostname + "_" + title
    msg['From'] = me
    msg['To'] = ",".join(you)

    # attach images
    for ff in image_files:
        with open(ff, 'rb') as fp:
            img = MIMEImage(fp.read())
            msg.attach(img)
    
    # attach texts
    for ff in txt_files:
        with open(ff, 'rb') as fp:
            txt = MIMEText(fp.read())
            msg.attach(txt)
    
    s = smtplib.SMTP(smtp_server)
    s.sendmail(me, you, msg.as_string())
    s.quit()

