# -*- coding: utf-8 -*-
from __future__ import absolute_import
from collections import OrderedDict

import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable
import scipy.optimize

from .chainer_utils import *

import gym


class Gaussian(object):
    def __init__(self, means, log_stds):
        self.means = means
        self.log_stds = log_stds

    def unchain_backward(self):
        self.means.unchain_backward()
        self.log_stds.unchain_backward()

    def as_dict(self):
        return dict(means=self.means, log_stds=self.log_stds)

    @classmethod
    def from_dict(cls, d):
        return cls(means=d["means"], log_stds=d["log_stds"])

    def sample(self):
        xp = chainer.cuda.get_array_module(self.means.data)
        zs = xp.random.normal(size=self.means.data.shape).astype(xp.float32)
        return self.means + chainer.Variable(zs) * F.exp(self.log_stds)
        # return F.gaussian(self.means, self.log_stds * 2)

    def logli(self, a):
        a = F.cast(a, np.float32)
        # transform back to standard normal
        zs = (a - self.means) * F.exp(-self.log_stds)

        # density of standard normal: f(z) = (2*pi*det|Σ|)^(-n/2) * exp(-|x|^2/2)
        # the return value should be log f(z)
        return - F.sum(self.log_stds, axis=-1) - \
               0.5 * F.sum(F.square(zs), axis=-1) - \
               0.5 * self.means.shape[-1] * np.log(2 * np.pi)

    def likelihood_ratio(self, other, a):
        """
        Compute p_self(a) / p_other(a)
        """
        logli = self.logli(a)
        other_logli = other.logli(a)
        return F.exp(logli - other_logli)

    def kl_div(self, other):
        """
        Given the distribution parameters of two diagonal multivariate Gaussians, compute their KL divergence (vectorized)

        Reference: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Kullback.E2.80.93Leibler_divergence_for_multivariate_normal_distributions

        In general, for two n-dimensional distributions, we have

        D_KL(N1||N2) = 1/2 ( tr(Σ_2^{-1}Σ_1) + (μ_2 - μ_1)^T Σ_2^{-1} (μ_2 - μ_1) - n + ln(det(Σ_2) / det(Σ_1)) )

        Here, Σ_1 and Σ_2 are diagonal. Hence this equation can be simplified. In terms of the parameters of this method,

            - ln(det(Σ_2) / det(Σ_1)) = sum(2 * (log_stds_2 - log_stds_1), axis=-1)

            - (μ_2 - μ_1)^T Σ_2^{-1} (μ_2 - μ_1) = sum((means_1 - means_2)^2 / vars_2, axis=-1)

            - tr(Σ_2^{-1}Σ_1) = sum(vars_1 / vars_2, axis=-1)

        Where

            - vars_1 = exp(2 * log_stds_1)

            - vars_2 = exp(2 * log_stds_2)

        Combined together, we have

        D_KL(N1||N2) = 1/2 ( tr(Σ_2^{-1}Σ_1) + (μ_2 - μ_1)^T Σ_2^{-1} (μ_2 - μ_1) - n + ln(det(Σ_2) / det(Σ_1)) )
                     = sum(1/2 * ((vars_1 - vars_2) / vars_2 + (means_1 - means_2)^2 / vars_2 + 2 * (log_stds_2 - log_stds_1)), axis=-1)
                     = sum( ((means_1 - means_2)^2 + vars_1 - vars_2) / (2 * vars_2) + (log_stds_2 - log_stds_1)), axis=-1)

        :param means_1: List of mean parameters of the first distribution
        :param log_stds_1: List of log standard deviation parameters of the first distribution
        :param means_2: List of mean parameters of the second distribution
        :param log_stds_2: List of log standard deviation parameters of the second distribution
        :return: An array of KL divergences.
        """

        vars = F.exp(2 * self.log_stds)
        other_vars = F.exp(2 * other.log_stds)

        return F.sum((F.square(self.means - other.means) + vars - other_vars) /
                     (2 * other_vars + 1e-8) + other.log_stds - self.log_stds, axis=-1)

    def entropy(self):
        return F.sum(self.log_stds + np.log(np.sqrt(2 * np.pi * np.e)), axis=-1)


class Model(chainer.Chain):
    def __init__(self, observation_space, action_space, env_spec, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.observation_space = observation_space
        self.action_space = action_space
        self.env_spec = env_spec

        # self.obs_dim = flatten_dim(observation_space)
        # self.action_dim = flatten_dim(action_space)
        self.obs_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]


class NNFeatureModel(Model):
    feature_dim = None

    def compute_features(self, obs):
        raise NotImplementedError

    def feature_links(self):
        raise NotImplementedError


class MLPFeatureModel(NNFeatureModel):
    def __init__(self, observation_space, action_space, env_spec, hidden_sizes=(128, 64), hidden_nonlinearity=F.relu,
                 **kwargs):
        super(MLPFeatureModel, self).__init__(observation_space, action_space, env_spec, **kwargs)
        self.hidden_sizes = hidden_sizes
        if isinstance(hidden_nonlinearity, str):
            if hidden_nonlinearity == 'relu':
                hidden_nonlinearity = F.relu
            elif hidden_nonlinearity == 'tanh':
                hidden_nonlinearity = F.tanh
            elif hidden_nonlinearity == 'elu':
                hidden_nonlinearity = F.elu
            else:
                raise NotImplementedError
        self.hidden_nonlinearity = hidden_nonlinearity
        self.n_layers = len(hidden_sizes)
        self._feature_links = OrderedDict()
        with self.init_scope():
            input_size = self.obs_dim
            for idx, hidden_size in enumerate(hidden_sizes):
                link = L.Linear(input_size, hidden_size)
                name = "fc{}".format(idx + 1)
                setattr(self, name, link)
                self._feature_links[name] = link
                input_size = hidden_size
            self.feature_dim = input_size

    def feature_links(self):
        return self._feature_links

    def compute_features(self, obs):
        obs = F.cast(obs, np.float32)
        h = obs
        for link in self.feature_links().values():
            h = self.hidden_nonlinearity(link(h))
        return h


class WeightSharingFeatureModel(NNFeatureModel):
    def __init__(self, observation_space, action_space, env_spec, feature_model, **kwargs):
        super(WeightSharingFeatureModel, self).__init__(observation_space, action_space, env_spec, **kwargs)
        with self.init_scope():
            for name, link in feature_model.feature_links().items():
                setattr(self, name, link)
        self._compute_features = feature_model.compute_features
        self.feature_dim = feature_model.feature_dim
        self.feature_model = feature_model
        self._feature_links = feature_model.feature_links

    def compute_features(self, obs):
        return self._compute_features(obs)

    def feature_links(self):
        return self._feature_links()


class ValueFunction(Model):
    def compute_state_values(self, obs):
        raise NotImplementedError


class NNFeatureValueFunction(ValueFunction, NNFeatureModel):
    def __init__(self, observation_space, action_space, env_spec, **kwargs):
        super(NNFeatureValueFunction, self).__init__(observation_space, action_space, env_spec, **kwargs)
        with self.init_scope():
            self.l_vf = L.Linear(self.feature_dim, 1)

    def compute_state_values(self, obs, feats=None):
        if feats is None:
            feats = super(NNFeatureValueFunction, self).compute_features(obs)
        return self.l_vf(feats)[..., 0]


class WeightSharingValueFunction(NNFeatureValueFunction, WeightSharingFeatureModel):
    pass


class Policy(Model):
    feature_dim = None

    def compute_dists(self, obs):
        """
        Given some observations, compute the parameters for the action distributions.
        :param obs: A chainer variable containing a list of observations.
        :return: An instance of the Distribution class, represeting the action distribution.
        """
        raise NotImplementedError

    def get_actions(self, obs):
        with chainer.no_backprop_mode():
            dists = self.compute_dists(Variable(np.asarray(obs)))
            actions = dists.sample()
            return actions.data, {k: v.data for k, v in dists.as_dict().items()}

    def get_action(self, ob):
        actions, dists = self.get_actions(np.expand_dims(ob, 0))
        return actions[0], {k: v[0] for k, v in dists.items()}


class NNFeaturePolicy(Policy, NNFeatureModel):
    def create_vf(self):
        return WeightSharingValueFunction(
            observation_space=self.observation_space,
            action_space=self.action_space,
            env_spec=self.env_spec,
            feature_model=self,
        )


class GaussianPolicy(NNFeaturePolicy):
    def __init__(self, observation_space, action_space, env_spec, **kwargs):
        super(GaussianPolicy, self).__init__(observation_space, action_space, env_spec, **kwargs)
        with self.init_scope():
            assert self.feature_dim is not None
            self.l_act = L.Linear(self.feature_dim, self.action_dim)
            self.log_std = chainer.Parameter(
                shape=(self.action_dim,), initializer=chainer.initializers.Zero())

    def compute_dists(self, obs, feats=None):
        if feats is None:
            feats = super(GaussianPolicy, self).compute_features(obs)
        means = self.l_act(feats)
        # for this policy, the variance is independent of the state
        log_stds = F.tile(self.log_std.reshape((1, -1)), (len(feats), 1))
        return Gaussian(means=means, log_stds=log_stds)

    @property
    def distribution(self):
        return Gaussian


class GaussianMLPPolicy(GaussianPolicy, MLPFeatureModel):
    pass


class Baseline(object):
    def __init__(self, *args, **kwargs):
        super(Baseline, self).__init__(*args, **kwargs)

    def predict(self, obs):
        raise NotImplementedError

    def update(self, trajs):
        raise NotImplementedError


class NNFeatureBaseline(Baseline, NNFeatureModel):
    def __init__(self, observation_space, action_space, env_spec, mixture_fraction=0.1, **kwargs):
        # For input, we will concatenate the observation with the time, so we need to increment the observation
        # dimension
        if isinstance(observation_space, gym.spaces.Box) and len(observation_space.shape) == 1:
            self.concat_time = True
            obs_space = gym.spaces.Box(
                low=np.append(observation_space.low, 0),
                high=np.append(observation_space.high, 2 ** 32),
            )
        else:
            obs_space = observation_space
        self.mixture_fraction = mixture_fraction
        super(NNFeatureBaseline, self).__init__(obs_space, action_space, env_spec, **kwargs)
        with self.init_scope():
            assert self.feature_dim is not None
            self.l_b = L.Linear(self.feature_dim, self.action_dim)

    def compute_baselines(self, obs):
        feats = self.compute_features(obs)
        return self.l_b(feats)[..., 0]

    # Value functions can themselves be used as baselines
    def predict(self, obs):
        with chainer.no_backprop_mode():
            if self.concat_time:
                ts = np.arange(len(obs)) / self.env_spec.timestep_limit
                obs = np.concatenate([obs, ts[:, None]], axis=-1)
            values = self.compute_baselines(Variable(obs))
            return values.data

    # By default, when used as baselines, value functions are updated via L-BFGS
    def update(self, trajs):
        obs = np.concatenate([traj['observations'] for traj in trajs], axis=0)
        if self.concat_time:
            ts = np.concatenate([np.arange(len(traj['observations'])) / self.env_spec.timestep_limit for traj in trajs],
                                axis=0)
            obs = np.concatenate([obs, ts[:, None]], axis=-1)
        returns = np.concatenate([traj['returns'] for traj in trajs], axis=0)
        baselines = np.concatenate([traj['baselines']
                                    for traj in trajs], axis=0)

        # regress to a mixture of current and past predictions
        targets = returns * (1. - self.mixture_fraction) + \
                  baselines * self.mixture_fraction

        # use lbfgs to perform the update
        cur_params = get_flat_params(self)

        obs = Variable(obs)
        targets = Variable(targets.astype(np.float32))

        def f_loss_grad(x):
            set_flat_params(self, x)
            self.cleargrads()
            values = self.compute_baselines(obs)
            loss = F.mean(F.square(values - targets))
            loss.backward()
            flat_grad = get_flat_grad(self)
            return loss.data.astype(np.float64), flat_grad.astype(np.float64)

        new_params = scipy.optimize.fmin_l_bfgs_b(
            f_loss_grad, cur_params, maxiter=10)[0]

        set_flat_params(self, new_params)


class MLPBaseline(NNFeatureBaseline, MLPFeatureModel):
    pass
