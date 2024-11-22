import os
import operator
import pdb

import jax
import jax.numpy as jnp
from jax import lax
from jax import random
import distrax
import numpy as np
from scipy.stats import ortho_group


def jittable_sim_lba(A_key, v_key, v, t0, b, A, K, s=1):
    """
    Generate RTs and choices from the LBA model using the provided parameters,
    with different drift rates for each trial (v).
    """
    k = random.uniform(A_key, shape=(K,), minval=0.0, maxval=A)  # starting points
    sigma = s * jnp.eye(K)
    drifts = random.multivariate_normal(v_key, v, sigma, shape=(1,))  # drifts
    # Find trials with drifts all negative and remove (very rare)
    neg_drifts = jnp.where(drifts < 0, 0, 1)
    valid_idx = jnp.where(jnp.sum(neg_drifts, axis=1) > 0, 1, 0)
    # Change negative drifts to small positive constant
    drifts = jnp.where(drifts > 0, drifts, 1e-6)
    all_t = (b - k) / drifts
    choices = jnp.argmin(all_t, 1)
    rts = jnp.min(all_t, 1)
    return (rts + t0).squeeze(), choices.squeeze(), valid_idx.squeeze()


def sim_lba_rts(v, A, b, t0, key, n, K, batch_drifts=None, prep_data=False):
    # Wrapper for jittable_sim_lba
    sim_keys = random.split(key, 2 * n)
    if batch_drifts is None:
        batch_sim_lba = jax.vmap(
            jittable_sim_lba, in_axes=[0, 0, None, None, None, None, None]
        )
        rts, resp, valid_idx = batch_sim_lba(sim_keys[:n], sim_keys[n:], v, t0, b, A, K)
    else:
        batch_sim_lba = jax.vmap(
            jittable_sim_lba, in_axes=[0, 0, 0, None, None, None, None]
        )
        rts, resp, valid_idx = batch_sim_lba(
            sim_keys[:n], sim_keys[n:], batch_drifts, t0, b, A, K
        )
    if prep_data:
        valid_idx = np.nonzero(valid_idx == 1)
        sim_data = {
            "response_dirs": resp[valid_idx],
            "rts": rts[valid_idx],
            "valid_idx": valid_idx[valid_idx],
        }
    else:
        sim_data = {
            "response_dirs": resp,
            "rts": rts,
            "valid_idx": valid_idx.astype(bool),
        }
    return sim_data


def generate_vam_rts(lba_params, drifts, n_acc, mc_key):
    sim_data = sim_lba_rts(
        None,
        lba_params["a"],
        lba_params["b"],
        lba_params["t0"],
        mc_key,
        len(drifts),
        n_acc,
        batch_drifts=drifts,
    )
    return sim_data


def lba_logp(t, c, v, b, A, t0, s):
    """
    Jittable/vmappable function that calculates the log-likelihood of a given
    RT (t) and choice (c) under the LBA model with params v, b, A, t0, and s.
    The choice (c) should be one-cold encoded (not a one-hot vector).
    The mean drift rate (v) is a vector, one for each possible choice.
    """

    trel = t - t0
    Ndist = distrax.Normal(0, 1)
    w1 = (b - trel * v[c]) / (trel * s)
    w2 = A / (trel * s)
    # Calculate fc(t)
    fc_p = (1 / A) * (
        -v[c] * _round_cdfs(Ndist.cdf(w1 - w2))
        + s * Ndist.prob(w1 - w2)
        + v[c] * _round_cdfs(Ndist.cdf(w1))
        - s * Ndist.prob(w1)
    )
    fc_logp = jnp.log(_fix_negprob(fc_p))
    # Calculate sum of 1-Fk(t) for each k ~= c
    Fterms_logp = _sum_CDF_terms(trel, c, v, b, A, s, Ndist)
    return fc_logp + Fterms_logp


def _single_CDF_term(trel, vk, b, A, s, Ndist):
    w1 = (b - trel * vk) / (trel * s)
    w2 = A / (trel * s)
    Fk_t = (
        1
        + (1 / A) * (b - A - trel * vk) * _round_cdfs(Ndist.cdf(w1 - w2))
        - (1 / A) * (b - trel * vk) * _round_cdfs(Ndist.cdf(w1))
        + (1 / w2) * Ndist.prob(w1 - w2)
        - (1 / w2) * Ndist.prob(w1)
    )
    return jnp.array([jnp.log(1 - Fk_t)])


def _sum_CDF_terms(trel, c, v, b, A, s, Ndist):
    kvec = jnp.arange(len(v))

    def step(carry, k):
        logp, trel, c, v, b, A, s, Ndist = carry
        p_step = jnp.array(
            [jnp.sum(jnp.array([logp, _single_CDF_term(trel, v[k], b, A, s, Ndist)]))]
        )
        return (p_step, trel, c, v, b, A, s, Ndist), p_step

    carry, _ = lax.scan(step, (jnp.array([0.0]), trel, c, v, b, A, s, Ndist), kvec)
    return carry[0] - _single_CDF_term(trel, v[c], b, A, s, Ndist)


def _round_cdfs(x, minval=0.001, maxval=0.999):
    # Numerical error in CDF values close to zero or close to one can
    # lead to negative probabilities in fc_logp and Fterms_logp,
    # so we round to avoid.
    x = jnp.where(x < minval, minval * jnp.ones(x.shape), x)
    x = jnp.where(x > maxval, maxval * jnp.ones(x.shape), x)
    return x


def _fix_negprob(x, minval=1e-40):
    # Adjust negative probabilities to a small positive value.
    return jnp.where(x <= 0, minval * jnp.ones(x.shape), x)
