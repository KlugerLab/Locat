from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import tensorflow_probability.substrates.jax as jaxp

jax.config.update("jax_enable_x64", True)

from sklearn.cluster import kmeans_plusplus
from loguru import logger

for d in jax.local_devices():
    logger.info(f'Found device: {d}')
jaxd = jaxp.distributions


def _weighted_kmeans_init(X, w, n_c, n_inits):
    return map(
        lambda i: kmeans_plusplus(X, n_clusters=n_c, sample_weight=w[:, i])[0],
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
def train_em(X, samples_weights, mu_init, sigma_init, n_components,
             n_inits=25, reg_covar=0.0, rtol=1e-6, max_iter=500, seed=1):
    
    def cond_fn(state):
        i, thetas, loss, loss_diff, _ = state
        return jnp.all((i < max_iter) & (loss_diff > rtol))

    @jax.vmap
    def one_step(state):
        i, (pi, mu, sigma), loss, loss_diff, sample_weights = state
        membership_weight = e_step(X, pi, mu, sigma)

        pi_updated, mu_updated, sigma_updated = weighted_m_step(X, membership_weight, sample_weights, reg_covar)
        loss_updated = compute_loss(
            X, pi_updated, mu_updated, sigma_updated, membership_weight)
        loss_diff = jnp.abs((loss_updated / loss) - 1.)

        return (i + 1,
                (pi_updated, mu_updated, sigma_updated),
                loss_updated,
                loss_diff,
                sample_weights)

    key = jax.random.PRNGKey(seed)
    raw_pi_init = jax.random.uniform(key, shape=(n_inits, n_components))
    pi_init = raw_pi_init / raw_pi_init.sum(-1, keepdims=True)
    key, subkey = jax.random.split(key)

    init_val = (jnp.zeros([n_inits], jnp.int32),
                (pi_init, mu_init, sigma_init),
                -jnp.ones([n_inits]) * jnp.inf,
                jnp.ones([n_inits]) * jnp.inf,
                samples_weights.T)

    num_iter, (pi_est, mu_est, sigma_est), loss, loss_diff, _ = jax.lax.while_loop(cond_fn, one_step, init_val)

    # index = jnp.argmax(loss)
    # pi_best, mu_best, sigma_best = jax.tree_map(lambda x: x[index], (pi_est, mu_est, sigma_est))

    return pi_est, mu_est, sigma_est


def softbootstrap_gmm(X, raw_weights, n_components, n_inits=100, reg_covar=0.0, seed=1, buckets=None):
    if buckets is None:
        buckets = np.clip(int(len(raw_weights)/30), 3, 30)
        
    rand_weights = raw_weights 
    o_fwd = np.argsort(raw_weights)
    o_back = np.argsort(o_fwd)

    rand_weights = np.repeat(rand_weights[o_fwd, None], n_inits, axis=1)
    rng = np.random.default_rng(seed)
    rand_weights = np.concatenate([rng.permuted(i, axis=0) for i in np.array_split(rand_weights, buckets, axis=0)], axis=0)
    rand_weights = rand_weights[o_back,:]

    n = rand_weights.shape[0]
    boot_weights = np.random.geometric(1 / n, size=rand_weights.shape)
    boot_weights = (boot_weights / np.sum(boot_weights, axis=0)[None, :])
    weights = rand_weights * boot_weights
    return rgmm(X, weights, n_components, n_inits, reg_covar, rand_weights)


def hardbootstrap_gmm(X, raw_weights, n_components, fraction, n_inits=30, reg_covar=0.0, seed=1):
    """
    fraction: what proportion of items to sample
    """
    norm_weights = raw_weights / np.sum(raw_weights)
    fraction = np.clip(fraction, 0, 1)
    n_points = X.shape[0]
    n_samples = np.maximum(1, int(n_points * fraction))

    weights = np.zeros(shape=(n_points, n_inits))
    for i in range(n_inits):
        sampled_indices = np.random.choice(
            n_points,
            size=n_samples,
            replace=True,
            p=norm_weights
        )
        i0, c0 = np.unique(sampled_indices, return_counts=True)
        weights[i0, i] = c0  # raw_weights[sampled_indices]

    return rgmm(X, weights, n_components, n_inits, reg_covar)


def simplebootstrap_gmm(X, n_components, fraction, n_inits=30, reg_covar=0.0, seed=1):
    """
    fraction: what proportion of items to sample
    """
    fraction = np.clip(fraction, 0, 1)
    n_points = X.shape[0]
    n_samples = np.maximum(1, int(n_points * fraction))

    weights = np.zeros(shape=(n_points, n_inits))
    for i in range(n_inits):
        sampled_indices = np.random.choice(
            n_points,
            size=n_samples,
            replace=False,
        )
        weights[sampled_indices, i] = 1/n_samples  # raw_weights[sampled_indices]

    return rgmm(X, weights, n_components, n_inits, reg_covar)


def rgmm(X, weights, n_components, n_inits, reg_covar, true_weights=None):
    if true_weights is None:
        true_weights = weights
    weights = weights / np.sum(weights)
    mu_init, sigma_init = weighted_gmm_init(X,
                                            w=weights,
                                            n_c=n_components,
                                            n_inits=n_inits)
    pi_est, mu_est, sigma_est = train_em(X,
                                         samples_weights=weights,
                                         mu_init=mu_init,
                                         sigma_init=sigma_init,
                                         n_components=n_components,
                                         n_inits=n_inits,
                                         reg_covar=reg_covar,
                                         rtol=1e-6,
                                         max_iter=500,
                                         seed=1)
    
    return np.array(pi_est), np.array(mu_est), np.array(sigma_est), true_weights


