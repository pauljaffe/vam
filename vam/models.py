import jax
import jax.numpy as jnp
from jax import random, lax, vmap
import flax
from flax import linen as nn
import distrax
import pdb

from .lba import lba_logp
from .utils import (
    constant_init,
    vec_to_lowertri,
    lowertri_to_vec,
    lba_jac_adj,
    batch_reparam_fr,
)


@flax.struct.dataclass
class ModelConfig:
    # Param initialization
    w_scale: float  # variational cov is intialized to w_scale*I
    a_init: float
    c_init: float
    t0_init: float
    s: float
    n_mc: int  # number of Monte Carlo samples
    conv_n_features: list  # number of features in each conv layer
    fc_n_units: list  # number of units in each fc layer
    dropout_rate: float
    param_dtype: jnp.dtype
    n_acc: int  # number of accumulators
    elbo_type: str
    test_only: bool
    model_type: str


class VAM(nn.Module):
    """Implementation of the visual accumulator model (VAM)."""

    config: ModelConfig
    param_dtype: jnp.dtype

    def setup(self):
        self.get_drifts = CNN(self.config, self.param_dtype)
        self.get_elbo = LBAVI(self.config, self.param_dtype)

    def __call__(self, stimuli, rts, responses, key, training: bool):
        drift_mean = self.get_drifts(stimuli, training=training)
        elbo = self.get_elbo(rts, responses, drift_mean, key)
        return elbo, drift_mean


class LBAVI(nn.Module):
    """Implementation of variational inference for the linear ballistic accumulator model (LBA)."""

    config: ModelConfig
    param_dtype: jnp.dtype

    @property
    def mu(self):
        return jnp.concatenate([self.c, self.a, self.t0])

    @property
    def lcov(self):
        return vec_to_lowertri(self.w, len(self.mu))

    def setup(self):
        config = self.config
        self.a = self.param("a", constant_init(config.a_init), (1,))
        self.c = self.param("c", constant_init(config.c_init), (1,))
        self.t0 = self.param("t0", constant_init(config.t0_init), (1,))

        # Covariance matrix for variational parameters
        self.w = self.param(
            "w",
            lambda key, shape: config.w_scale
            * lowertri_to_vec(jnp.eye(len(self.mu)), len(self.mu)),
            (1,),
        )

    def _reparameterize(self, batch_size, key):
        # Reparameterization trick
        config = self.config
        if config.elbo_type == "standard":
            keys = random.split(key, config.n_mc)
            rsample = batch_reparam_fr(keys, self.mu, self.lcov)
            rsample = jnp.reshape(rsample, (config.n_mc, len(self.mu)))
        elif config.elbo_type == "local":
            keys = random.split(key, batch_size * config.n_mc)
            rsample = batch_reparam_fr(keys, self.mu, self.lcov)
            rsample = jnp.reshape(rsample, (batch_size, config.n_mc, len(self.mu)))
        return rsample

    def elbo(self, rts, choices, drifts, key):
        if self.config.elbo_type == "standard":
            get_jac = vmap(lba_jac_adj, in_axes=[0, 0, 0])
            get_logp = jax.vmap(
                jax.vmap(lba_logp, in_axes=[0, 0, 0, None, None, None, None]),
                in_axes=[None, None, None, 0, 0, 0, None],
            )
        elif self.config.elbo_type == "local":
            get_jac = vmap(vmap(lba_jac_adj, in_axes=[0, 0, 0]), in_axes=[0, 0, 0])
            get_logp = jax.vmap(
                jax.vmap(lba_logp, in_axes=[None, None, None, 0, 0, 0, None]),
                in_axes=[0, 0, 0, 0, 0, 0, None],
            )

        # Sample and reparameterize
        rsample = self._reparameterize(len(rts), key)
        c = jnp.exp(rsample[..., 0])
        a = jnp.exp(rsample[..., 1])
        t0 = jnp.exp(rsample[..., 2])
        b = c + a

        # Reconstruction error term
        log_px_z = get_logp(rts, choices, drifts, b, a, t0, self.config.s).squeeze()

        # Jacobian determinant term
        jac_adj = get_jac(
            rsample[..., 1],
            rsample[..., 0],
            rsample[..., 2],
        )

        # Entropy term
        q_dist = distrax.MultivariateNormalFullCovariance(
            self.mu, self.lcov @ self.lcov.T
        )
        log_q_z = q_dist.log_prob(rsample)

        # Prior term
        p_z_dist = distrax.MultivariateNormalDiag(
            jnp.zeros(len(self.mu)), jnp.ones(len(self.mu))
        )
        log_p_z = p_z_dist.log_prob(jnp.exp(rsample))

        if self.config.elbo_type == "standard":
            elbo = jnp.mean(
                jnp.sum(log_px_z, axis=1) + jac_adj + log_p_z - log_q_z
            ) / len(rts)
        elif self.config.elbo_type == "local":
            elbo = jnp.mean(
                jnp.sum(log_px_z, axis=0)
                + jnp.mean(jac_adj, axis=0)
                + jnp.mean(log_p_z, axis=0)
                - jnp.mean(log_q_z, axis=0)
            ) / len(rts)

        return elbo

    def __call__(self, rts, choices, drifts, key):
        return self.elbo(rts, choices, drifts, key)


class CNN(nn.Module):
    config: ModelConfig
    param_dtype: jnp.dtype

    @nn.compact
    def __call__(self, x, training: bool):
        for n in self.config.conv_n_features:
            x = nn.Conv(features=n, kernel_size=(3, 3), padding="same")(x)
            x = nn.relu(x)
            self.sow("intermediates", "features", x)
            x = nn.GroupNorm(num_groups=n)(x)
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten

        for n in self.config.fc_n_units:
            x = nn.Dense(features=n)(x)
            x = nn.relu(x)
            x = nn.Dropout(rate=self.config.dropout_rate, deterministic=not training)(x)
            self.sow("intermediates", "features", x)

        x = nn.Dense(features=self.config.n_acc)(x)

        return x
