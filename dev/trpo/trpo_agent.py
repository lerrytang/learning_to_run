# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np
import multiprocessing as mp

import chainer
import chainer.functions as F

from .chainer_utils import get_flat_params, set_flat_params, get_flat_grad, cg
from .models import GaussianMLPPolicy, MLPBaseline
# import ipdb


def fvp(policy, f_kl, grad0, v, eps=1e-5, damping=1e-8):
    """
    Approximately compute the Fisher-vector product of the provided policy, F(x)v, where x is the current policy parameter
    and v is the vector we want to form product with.

    Define g(x) to be the gradient of the KL divergence (f_kl) evaluated at x. Note that for small \\epsilon, Taylor expansion gives
    g(x + \\epsilon v) ≈ g(x) + \\epsilon F(x)v
    So
    F(x)v \\approx (g(x + \epsilon v) - g(x)) / \\epsilon
    Since x is always the current parameters, we cache the computation of g(x) and this is provided as an input, grad0

    :param policy: The policy to compute Fisher-vector product.
    :param f_kl: A function which computes the average KL divergence.
    :param grad0: The gradient of KL divergence evaluated at the current parameter x.
    :param v: The vector we want to compute product with.
    :param eps: A small perturbation for finite difference computation.
    :param damping: A small damping factor to ensure that the Fisher information matrix is positive definite.
    :return:
    """

    flat_params = get_flat_params(policy)

    # compute g(x + \epsilon v)
    set_flat_params(policy, flat_params + eps * v)
    policy.cleargrads()
    f_kl().backward()
    grad_plus = get_flat_grad(policy)

    # don't forget to restore the policy parameters!
    set_flat_params(policy, flat_params)

    # form the finite difference
    return (grad_plus - grad0) / (eps) + damping * flat_params


def linesearch(f, x0, dx, expected_improvement, y0=None, backtrack_ratio=0.8, max_backtracks=15, accept_ratio=0.1,
               atol=1e-7):
    """
    Perform line search on the function f at x, where
    :param f: The function to perform line search on.
    :param x0: The current parameter value.
    :param dx: The full descent direction. We will shrink along this direction.
    :param y0: The initial value of f at x (optional).
    :param backtrack_ratio: Ratio to shrink the descent direction per line search step.
    :param max_backtracks: Maximum number of backtracking steps
    :param expected_improvement: Expected amount of improvement when taking the full descent direction dx, typically
           computed by y0 - y \\approx (f_x|x=x0).dot(dx), where f_x|x=x0 is the gradient of f w.r.t. x, evaluated at x0.
    :param accept_ratio: minimum acceptance ratio of actual_improvement / expected_improvement
    :return: The descent step obtained
    """
    if expected_improvement >= atol:
        if y0 is None:
            y0 = f(x0)
        for ratio in backtrack_ratio ** np.arange(max_backtracks):
            x = x0 - ratio * dx
            y = f(x)
            actual_improvement = y0 - y
            if actual_improvement / (expected_improvement * ratio) >= accept_ratio:
                # logger.logkv("ExpectedImprovement",
                #              expected_improvement * ratio)
                # logger.logkv("ActualImprovement", actual_improvement)
                # logger.logkv("ImprovementRatio", actual_improvement /
                #              (expected_improvement * ratio))
                return x
    # logger.logkv("ExpectedImprovement", expected_improvement)
    # logger.logkv("ActualImprovement", 0.)
    # logger.logkv("ImprovementRatio", 0.)
    return x0


class TRPO(object):
    """
    This method implements Trust Region Policy Optimization. Without the line search step, this algorithm is equivalent
    to an approximate procedure for computing natural gradient using conjugate gradients, where it performs approximate
    Hessian-vector product computation using finite differences.

    :param env: An environment instance, which should have the same class as what env_maker.make() returns.
    :param env_maker: An object such that calling env_maker.make() will generate a new environment.
    :param policy: A stochastic policy which we will be optimizing.
    :param baseline: A baseline used for variance reduction and estimating future returns for unfinished trajectories.
    :param n_envs: Number of environments running simultaneously.
    :param last_iter: The index of the last iteration. This is normally -1 when starting afresh, but may be different when
           loaded from a snapshot.
    :param n_iters: The total number of iterations to run.
    :param batch_size: The number of samples used per iteration.
    :param discount: Discount factor.
    :param gae_lambda: Lambda parameter used for generalized advantage estimation. For details see the following paper:
    :param step_size: The maximum value of average KL divergence allowed per iteration.
    :param use_linesearch: Whether to perform line search using the surrogate loss derived in the TRPO algorithm.
           Without this step, the algorithm is equivalent to an implementation of natural policy gradient where we use
           conjugate gradient algorithm to approximately compute F^{-1}g, where F is the Fisher information matrix, and
           g is the policy gradient.
    :param kl_subsamp_ratio: The ratio we use to subsample data in computing the Hessian-vector products. This can
           potentially save a lot of time.
    :param snapshot_saver: An object for saving snapshots.
    """

    def __init__(self, env, env_maker, n_envs=mp.cpu_count(), last_iter=-1, n_iters=100,
                 batch_size=1000, discount=0.99, gae_lambda=0.97, step_size=0.01, use_linesearch=True,
                 kl_subsamp_ratio=1., snapshot_saver=None):
        self.env = env
        self.env_maker = env_maker
        self.n_envs = n_envs
        self.last_iter = last_iter
        self.n_iters = n_iters

        self.batch_size = batch_size

        self.discount = discount
        self.gae_lambda = gae_lambda
        self.step_size = step_size
        self.use_linesearch = use_linesearch
        self.kl_subsamp_ratio = kl_subsamp_ratio

        self.snapshot_saver = snapshot_saver

        self.policy = GaussianMLPPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            env_spec=env.spec,
            hidden_sizes=(256, 64),
            hidden_nonlinearity=chainer.functions.tanh,
        )
        self.baseline = MLPBaseline(
            observation_space=env.observation_space,
            action_space=env.action_space,
            env_spec=env.spec,
            hidden_sizes=(256, 64),
            hidden_nonlinearity=chainer.functions.tanh,
        )

    def compute_cumulative_returns(self, rewards, baselines, discount):
        # This method builds up the cumulative sum of discounted rewards for each time step:
        # R[t] = sum_{t'>=t} γ^(t'-t)*r_t'
        # Note that we use γ^(t'-t) instead of γ^t'. This gives us a biased gradient but lower variance
        returns = []
        # Use the last baseline prediction to back up
        cum_return = baselines[-1]
        for reward in rewards[::-1]:
            cum_return = cum_return * discount + reward
            returns.append(cum_return)
        return returns[::-1]

    def compute_advantages(self, rewards, baselines, discount, gae_lambda):
        # Given returns R_t and baselines b(s_t), compute (generalized) advantage estimate A_t
        deltas = rewards + discount * baselines[1:] - baselines[:-1]
        advs = []
        cum_adv = 0
        multiplier = discount * gae_lambda
        for delta in deltas[::-1]:
            cum_adv = cum_adv * multiplier + delta
            advs.append(cum_adv)
        return advs[::-1]

    def compute_pg_vars(self, trajs):
        """
        Compute chainer variables needed for various policy gradient algorithms
        """
        for traj in trajs:
            # Include the last observation here, in case the trajectory is not finished
            baselines = self.baseline.predict(np.concatenate([traj["observations"], [traj["last_observation"]]]))
            if traj['finished']:
                # If already finished, the future cumulative rewards starting from the final state is 0
                baselines[-1] = 0.
            # This is useful when fitting baselines. It uses the baseline prediction of the last state value to perform
            # Bellman backup if the trajectory is not finished.
            traj['returns'] = self.compute_cumulative_returns(traj['rewards'], baselines, self.discount)
            traj['advantages'] = self.compute_advantages(traj['rewards'], baselines, self.discount, self.gae_lambda)
            traj['baselines'] = baselines[:-1]

        # First, we compute a flattened list of observations, actions, and advantages
        all_obs = np.concatenate([traj['observations'] for traj in trajs], axis=0)
        all_acts = np.concatenate([traj['actions'] for traj in trajs], axis=0)
        all_advs = np.concatenate([traj['advantages'] for traj in trajs], axis=0)
        all_dists = {
            k: np.concatenate([traj['distributions'][k] for traj in trajs], axis=0)
            for k in trajs[0]['distributions'].keys()
        }

        # Normalizing the advantage values can make the algorithm more robust to reward scaling
        all_advs = (all_advs - np.mean(all_advs)) / (np.std(all_advs) + 1e-8)

        # Form chainer variables
        all_obs = chainer.Variable(all_obs)
        all_acts = chainer.Variable(all_acts)
        all_advs = chainer.Variable(all_advs.astype(np.float32, copy=False))
        all_dists = self.policy.distribution.from_dict(
            {k: chainer.Variable(v) for k, v in all_dists.items()})

        return all_obs, all_acts, all_advs, all_dists

    def parallel_collect_samples(self, env_pool):
        """
        Collect trajectories in parallel using a pool of workers. Actions are computed using the provided policy.
        Collection will continue until at least num_samples trajectories are collected. It will exceed this amount by
        at most env_pool.n_envs. This means that some of the trajectories will not be executed until termination. These
        partial trajectories will have their "finished" entry set to False.

        When starting, it will first check if env_pool.last_obs is set, and if so, it will start from there rather than
        resetting all environments. This is useful for reusing the same episode.

        :param env_pool: An instance of EnvPool.
        :param num_samples: The minimum number of samples to collect.
        :return:
        """
        trajs = []
        partial_trajs = [None] * env_pool.n_envs
        num_collected = 0

        if env_pool.last_obs is not None:
            obs = env_pool.last_obs
        else:
            obs = env_pool.reset()

        # if logger.get_level() <= logger.INFO:
        #     progbar = tqdm(total=num_samples)
        # else:
        #     progbar = None

        while num_collected < self.batch_size:
            actions, dists = self.policy.get_actions(obs)
            next_obs, rews, dones, infos = env_pool.step(actions)
            for idx in range(env_pool.n_envs):
                if partial_trajs[idx] is None:
                    partial_trajs[idx] = dict(
                        observations=[],
                        actions=[],
                        rewards=[],
                        distributions=[],
                    )
                traj = partial_trajs[idx]
                traj["observations"].append(obs[idx])
                traj["actions"].append(actions[idx])
                traj["rewards"].append(rews[idx])
                traj_dists = traj["distributions"]
                traj_dists.append({k: v[idx] for k, v in dists.items()})
                if dones[idx]:
                    trajs.append(
                        dict(
                            observations=np.asarray(traj["observations"]),
                            actions=np.asarray(traj["actions"]),
                            rewards=np.asarray(traj["rewards"]),
                            distributions={
                                k: np.asarray([d[k] for d in traj_dists])
                                for k in traj_dists[0].keys()
                            },
                            last_observation=infos[idx]["last_observation"],
                            finished=True,
                        )
                    )
                    partial_trajs[idx] = None
            obs = next_obs
            num_collected += env_pool.n_envs
        # if progbar is not None:
        #         progbar.update(env_pool.n_envs)
        #
        # if progbar is not None:
        #     progbar.close()

        for idx in range(env_pool.n_envs):
            if partial_trajs[idx] is not None:
                traj = partial_trajs[idx]
                traj_dists = traj["distributions"]
                trajs.append(
                    dict(
                        observations=np.asarray(traj["observations"]),
                        actions=np.asarray(traj["actions"]),
                        rewards=np.asarray(traj["rewards"]),
                        distributions={
                            k: np.asarray([d[k] for d in traj_dists])
                            for k in traj_dists[0].keys()
                        },
                        last_observation=obs[idx],
                        finished=False,
                    )
                )

        return trajs

    # def collect_samples(self, batch_size):
    def learn(self):

        # logger.info("Starting env pool asdasdfasdfasdf")

        with EnvPool(self.env_maker, n_envs=self.n_envs) as env_pool:
            for iter in range(self.last_iter + 1, self.n_iters):
                # logger.info("Starting iteration {}".format(iter))
                # logger.logkv('Iteration', iter)
                # logger.info("Start collecting samples")
                trajs = self.parallel_collect_samples(env_pool)
                # logger.info("Computing input variables for policy optimization")
                all_obs, all_acts, all_advs, all_dists = self.compute_pg_vars(trajs)

                # logger.info("Performing policy update")

                # subsample for kl divergence computation
                mask = np.zeros(len(all_obs), dtype=np.bool)
                mask_ids = np.random.choice(len(all_obs), size=int(
                    np.ceil(len(all_obs) * self.kl_subsamp_ratio)), replace=False)
                mask[mask_ids] = 1
                if self.kl_subsamp_ratio < 1:
                    subsamp_obs = all_obs[mask]
                    subsamp_dists = self.policy.distribution.from_dict(
                        {k: v[mask] for k, v in all_dists.as_dict().items()})
                else:
                    subsamp_obs = all_obs
                    subsamp_dists = all_dists

                # Define helper functions to compute surrogate loss and/or KL divergence. They share part of the computation
                # graph, so we use a common function to decide whether we should compute both (which is needed in the line
                # search phase)
                def f_loss_kl_impl(need_loss, need_kl):
                    retval = dict()
                    if need_loss:
                        new_dists = self.policy.compute_dists(all_obs)
                        old_dists = all_dists
                    elif need_kl:
                        # if only kl is needed, compute distribution from sub-sampled data
                        new_dists = self.policy.compute_dists(subsamp_obs)
                        old_dists = subsamp_dists
                    else:
                        raise ValueError('something wrong')

                    def compute_surr_loss(old_dists, new_dists, all_acts, all_advs):
                        ratio = new_dists.likelihood_ratio(old_dists, all_acts)
                        surr_loss = -F.mean(ratio * all_advs)
                        return surr_loss

                    def compute_kl(old_dists, new_dists):
                        kl = F.mean(old_dists.kl_div(new_dists))
                        return kl

                    if need_loss:
                        retval["surr_loss"] = compute_surr_loss(old_dists, new_dists, all_acts, all_advs)
                    if need_kl:
                        retval["kl"] = compute_kl(old_dists, new_dists)
                    return retval

                def f_surr_loss():
                    return f_loss_kl_impl(need_loss=True, need_kl=False)["surr_loss"]

                def f_kl():
                    return f_loss_kl_impl(need_loss=False, need_kl=True)["kl"]

                def f_surr_loss_kl():
                    retval = f_loss_kl_impl(need_loss=True, need_kl=True)
                    return retval["surr_loss"], retval["kl"]

                # Step 1: compute gradient in Euclidean space

                # logger.info("Computing gradient in Euclidean space")

                self.policy.cleargrads()

                surr_loss = f_surr_loss()
                surr_loss.backward()

                # obtain the flattened gradient vector
                flat_grad = get_flat_grad(self.policy)

                # Optimize memory usage: after getting the gradient, we do not need the rest of the computation graph
                # anymore
                surr_loss.unchain_backward()

                # Step 2: Perform conjugate gradient to compute approximate natural gradient

                # logger.info("Computing approximate natural gradient using conjugate gradient algorithm")

                self.policy.cleargrads()

                f_kl().backward()
                flat_kl_grad = get_flat_grad(self.policy)

                def Fx(v):
                    return fvp(self.policy, f_kl, flat_kl_grad, v)

                descent_direction = cg(Fx, flat_grad)

                # Step 3: Compute initial step size

                # We'd like D_KL(old||new) <= step_size
                # The 2nd order approximation gives 1/2*d^T*H*d <= step_size, where d is the descent step
                # Hence given the initial direction d_0 we can rescale it so that this constraint is tight
                # Let this scaling factor be \alpha, i.e. d = \alpha*d_0
                # Solving 1/2*\alpha^2*d_0^T*H*d_0 = step_size we get \alpha = \sqrt(2 * step_size / d_0^T*H*d_0)

                scale = np.sqrt(
                    2.0 * self.step_size *
                    (1. / (descent_direction.dot(Fx(descent_direction)) + 1e-8))
                )

                descent_step = descent_direction * scale

                cur_params = get_flat_params(self.policy)

                if self.use_linesearch:
                    # Step 4: Perform line search

                    # logger.info("Performing line search")

                    expected_improvement = flat_grad.dot(descent_step)

                    def f_barrier(x):
                        set_flat_params(self.policy, x)
                        with chainer.no_backprop_mode():
                            surr_loss, kl = f_surr_loss_kl()
                        return surr_loss.data + 1e100 * max(kl.data - self.step_size, 0.)

                    new_params = linesearch(
                        f_barrier,
                        x0=cur_params,
                        dx=descent_step,
                        y0=surr_loss.data,
                        expected_improvement=expected_improvement
                    )

                else:
                    new_params = cur_params - descent_step

                set_flat_params(self.policy, new_params)

                # logger.info("Updating baseline")
                self.baseline.update(trajs)

                # log statistics
                # logger.info("Computing logging information")
                with chainer.no_backprop_mode():
                    mean_kl = f_kl().data

                # logger.logkv('MeanKL', mean_kl)
                # log_action_distribution_statistics(all_dists)
                # log_reward_statistics(env)
                # log_baseline_statistics(trajs)
                # logger.dumpkvs()

                if self.snapshot_saver is not None:
                    # logger.info("Saving snapshot")
                    self.snapshot_saver.save_state(
                        iter,
                        dict(
                            alg=self,
                            alg_state=dict(
                                env_maker=self.env_maker,
                                policy=self.policy,
                                baseline=self.baseline,
                                n_envs=self.n_envs,
                                last_iter=iter,
                                n_iters=self.n_iters,
                                batch_size=self.batch_size,
                                discount=self.discount,
                                gae_lambda=self.gae_lambda,
                                step_size=self.step_size,
                                use_linesearch=self.use_linesearch,
                                kl_subsamp_ratio=self.kl_subsamp_ratio,
                            )
                        )
                    )
