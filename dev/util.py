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


def print_episode_info(logger, episode_info, pid):
    episode_n = episode_info["episode"]
    episode_steps = episode_info["steps"]
    episode_reward = episode_info["total_reward"]
    episode_loss = episode_info["loss"]
    episode_qval = episode_info["qval"]
    logger.info("pid={0}, episode={1}, steps={2}, rewards={3:.4f}, avg_loss={4:.4f}, avg_q={5:.4f}".format(
        pid, episode_n, episode_steps, episode_reward, np.mean(episode_loss), np.mean(episode_qval)))


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


def send_email(title, image_files, txt_files, smtp_server=None):

    if smtp_server is None:
        return

    me = "drl_search@XXX"
    you = ["lerrytang@gmail.com"]
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

