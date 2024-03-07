import os
import pickle
import pdb

import jax.numpy as jnp
from jax import random, vmap
import numpy as np
from flax import traverse_util
import scipy
import pandas as pd
import wandb
import matplotlib.pyplot as plt


def plot_batch_imgs(x, y, figsize=(12, 8), num_rows=5, num_columns=3, title=None):
    """Plots a batch of images x with labels y."""

    if len(x) != len(y):
        raise ValueError("Number of images and number of labels don't match!")

    _, ax = plt.subplots(num_rows, num_columns, figsize=figsize)

    for i in range(num_rows * num_columns):
        try:
            img = x[i]
            label = str(y[i])
            ax[i // num_columns, i % num_columns].imshow(img)
            ax[i // num_columns, i % num_columns].set_title(label)
            ax[i // num_columns, i % num_columns].axis("off")
        except:
            pass

    plt.tight_layout()
    plt.show()


def constant_init(value, dtype="float32"):
    """Initialize flax linen module parameters to value (a constant)."""

    def _init(key, shape, dtype=dtype):
        return value * jnp.ones(shape, dtype)

    return _init


def reparam_fr(key, mu, L):
    """Reparameterization trick for a full-rank Gaussian parameterized by a
    lower triangular covariance matrix L and mean vector mu."""
    sample = random.multivariate_normal(key, jnp.zeros(len(mu)), jnp.eye(len(mu)))
    return mu + L @ sample


batch_reparam_fr = vmap(reparam_fr, in_axes=[0, None, None])


def vec_to_lowertri(v, D):
    """Transform vector v to a DxD lower triangular matrix."""
    tril_idx = jnp.tril_indices(D)
    L = jnp.zeros((D, D))
    return L.at[tril_idx].set(v)


def lowertri_to_vec(L, D):
    """Transform DxD lower triangular matrix L to a vector."""
    return L[jnp.tril_indices(D)]


def lba_jac_adj(A, c, t0):
    """Jacobian adjustment for the VAM."""
    return jnp.log(jnp.abs(A * c * t0))


def vam_label_fn(path, x):
    if path[0] == "get_drifts":
        return "cnn"
    else:
        return "vi"


def flattened_traversal(fn):
    def mask(tree):
        flat = traverse_util.flatten_dict(tree)
        return traverse_util.unflatten_dict({k: fn(k, v) for k, v in flat.items()})

    return mask


def get_vam_lba_params(state, save_path=None):
    c = jnp.exp(state.params["get_elbo"]["c"])
    a = jnp.exp(state.params["get_elbo"]["a"])
    b = c + a
    t0 = jnp.exp(state.params["get_elbo"]["t0"])
    params = {"a": a, "b": b, "t0": t0}

    if save_path is not None:
        params = {k: np.array(v) for k, v in params.items()}
        with open(save_path, "wb") as f:
            pickle.dump(params, f)

    return params


def get_wandb_info(entity, project):
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")

    user_list, expt_id_list = [], []
    for rr in runs:
        try:
            user_list.append(rr.name[4:])
            expt_id_list.append(rr.id)
        except:
            continue

    info_df = pd.DataFrame({"user_id": user_list, "expt_id": expt_id_list})

    return info_df
