from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import tensorflow_probability.substrates.jax as jaxp

#from jax.config import config as jax_config

from concurrent.futures import ThreadPoolExecutor

jax.config.update("jax_enable_x64", True)

from sklearn.cluster import kmeans_plusplus
from loguru import logger

for d in jax.local_devices():
    logger.info(f'Found device: {d}')
jaxd = jaxp.distributions


def _weighted_kmeans_init(X, w, n_c, n_inits):
    return map(
        lambda i: kmeans_plusplus(X[i::n_inits, ...], n_clusters=n_c, sample_weight=w[i::n_inits])[0],
        np.arange(n_inits),)


def weighted_gmm_init(X, w, n_c, n_inits):
    n = X.shape[-1]
    mu_init = jnp.array(list(_weighted_kmeans_init(X, w, n_c, n_inits)))
    sigma_init = jnp.tile(jnp.eye(n)[None, ...], (n_inits, n_c, 1, 1))
    return mu_init, sigma_init


@jax.jit
def e_step(X, pi, mu, sigma):
    mixture_log_prob = jaxd.MultivariateNormalTriL(
        loc=mu,
        scale_tril=jnp.linalg.cholesky(sigma)
    ).log_prob(X[:, None, ...]) + jnp.log(pi)
    log_membership_weight = mixture_log_prob - jsp.special.logsumexp(mixture_log_prob, axis=-1, keepdims=True)
    return jnp.exp(log_membership_weight)


@jax.jit
def weighted_m_step(X, membership_weight, sample_weights, reg_covar):
    n, m = X.shape
    w = membership_weight * sample_weights[..., None]
    w_sum = w.sum(0)
    pi_updated = w_sum / n
    pi_updated /= np.sum(pi_updated)
    w = w / w_sum

    mu_updated = jnp.sum(
        X[:, None, ...] * w[..., None],
        axis=0)

    centered_x = X[:, None, ...] - mu_updated

    sigma_updated = jnp.sum(
        jnp.einsum('...i,...j->...ij', centered_x, centered_x) *
        w[..., None, None],
        axis=0)

    sigma_updated = sigma_updated + jnp.diag(jnp.ones(shape=(m,))*reg_covar)[None, :]

    return pi_updated, mu_updated, sigma_updated


@jax.jit
def compute_loss(X, pi, mu, sigma, membership_weight):
    component_log_prob = jaxd.MultivariateNormalTriL(
        loc=mu,
        scale_tril=jnp.linalg.cholesky(sigma)
    ).log_prob(X[:, None, ...])

    loss = membership_weight * (
            jnp.log(pi) + component_log_prob - jnp.log(
        jnp.clip(membership_weight,
                 a_min=jnp.finfo(np.float64).eps)))
    return jnp.sum(loss)


@partial(jax.jit, static_argnums=(4, 5, 6, 7, 8, 9))
def train_em(X, sample_weights, mu_init, sigma_init, n_components,
             n_inits=25, reg_covar=0.0, rtol=1e-6, max_iter=500, seed=1):
    def cond_fn(state):
        i, thetas, loss, loss_diff = state
        return jnp.all((i < max_iter) & (loss_diff > rtol))

    @jax.vmap
    def one_step(state):
        i, (pi, mu, sigma), loss, loss_diff = state
        membership_weight = e_step(X, pi, mu, sigma)

        pi_updated, mu_updated, sigma_updated = weighted_m_step(X, membership_weight, sample_weights, reg_covar)
        loss_updated = compute_loss(
            X, pi_updated, mu_updated, sigma_updated, membership_weight)
        loss_diff = jnp.abs((loss_updated / loss) - 1.)

        return (i + 1,
                (pi_updated, mu_updated, sigma_updated),
                loss_updated,
                loss_diff)

    key = jax.random.PRNGKey(seed)
    raw_pi_init = jax.random.uniform(key, shape=(n_inits, n_components))
    pi_init = raw_pi_init / raw_pi_init.sum(-1, keepdims=True)
    key, subkey = jax.random.split(key)

    init_val = (jnp.zeros([n_inits], jnp.int32),
                (pi_init, mu_init, sigma_init),
                -jnp.ones([n_inits]) * jnp.inf,
                jnp.ones([n_inits]) * jnp.inf)

    num_iter, (pi_est, mu_est, sigma_est), loss, loss_diff = jax.lax.while_loop(cond_fn, one_step, init_val)

    index = jnp.argmax(loss)
    pi_best, mu_best, sigma_best = jax.tree.map(lambda x: x[index], (pi_est, mu_est, sigma_est))

    return pi_best, mu_best, sigma_best, loss


def wgmm(X, raw_weights, n_components, n_inits=1, reg_covar=0.0):
    norm_weights = raw_weights / np.sum(raw_weights)
    mu_init, sigma_init = weighted_gmm_init(X,
                                            w=norm_weights,
                                            n_c=n_components,
                                            n_inits=n_inits)
    return train_em(X,
                    sample_weights=norm_weights,
                    mu_init=mu_init,
                    sigma_init=sigma_init,
                    n_components=n_components,
                    n_inits=n_inits,
                    reg_covar=reg_covar,
                    rtol=1e-6,
                    max_iter=500,
                    seed=1)

