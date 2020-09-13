import functools

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

import scipy.optimize

import zotbin.reweight


@jax.jit
def softmax_transform(p):
    """Transform (n, m) arbitrary floats into (n, m) values in the range (0,1) with normalized columns
    using the softmax transform.
    """
    w = jnp.exp(p)
    return w / jnp.sum(w, axis=0)


def extend_vector(p):
    """Extend n-1 arbitrary floats into n values in the range (0,1) that sum to one.
    Based on https://stackoverflow.com/questions/3589214/generate-random-numbers-summing-to-a-predefined-value
    """
    wsort = jnp.sort(1 / (1 + jnp.exp(-p)))
    return jnp.concatenate((wsort[0:1], jnp.diff(wsort), 1 - wsort[-1:]))

# Extend (n-1, m) arbitrary floats into (n, m) values in the range (0,1) with normalized columns.
# Provides an alternative to softmax that eliminates softmax's degeneracy wrt p -> p + const
# and reduces the parameter space dimensionality accordingly.
extend_transform = jax.jit(jax.vmap(extend_vector, in_axes=1, out_axes=1))


@functools.partial(jax.jit, static_argnums=(1,))
def metrics(params, transform_fun, mixing_matrix, init_data, gals_per_arcmin2, fsky):
    """Calculate the 3x2 metrics for the specified parameters.
    Assertions are commented out below to allow JIT compilation.
    All optimizers below use this entry point for their objective function.
    """
    ##assert mixing_matrix.sum() == 1
    # Transform the unbounded input parameters to weights normalized over columns.
    transformed = transform_fun(params)
    ##assert np.all(transformed.sum(axis=0) == 1)
    # Apply the mixing matrix to obtain values of 1/N dn[i]/dz[j].
    dndz = jnp.dot(transformed, mixing_matrix)
    ##assert dndz.sum() == 1
    # Use the same redshift bins for both probes.
    probe_dndz = jnp.array([dndz, dndz])
    # Calculate the 3x2 SNR, FOM, FOM_DETF.
    return zotbin.reweight.reweighted_metrics(probe_dndz, *init_data[1:], gals_per_arcmin2, fsky)


def jax_optimize(optimizer, initial_params, transform_fun, mixing_matrix, init_data,
                 gals_per_arcmin2, fsky, metric, callback, nsteps=100):
    """Optimize weights wrt parameterized dndz weights using jax.

    Should normally be used via the driver :func:`optimize`.

    Uses a fast calculation of the metrics governed by init_data.

    Runs exactly nsteps and returns the max_score obtained and the corresponding best
    parameters and score history as numpy arrays.  Note that the best score is not
    necessarily from the last step.

    The input optimizer can any instantiated method from jax.experimental.optimizers,
    e.g. ``optimizers.sgd(learning_rate)``.
    """
    def score(p):
        return metrics(p, transform_fun, mixing_matrix, init_data,  gals_per_arcmin2, fsky)[metric]

    score_and_grads = jax.jit(jax.value_and_grad(score))

    opt_init, opt_update, opt_params = optimizer
    opt_state = opt_init(jnp.asarray(initial_params))

    @jax.jit
    def take_step(stepnum, opt_state):
        score, grads = score_and_grads(opt_params(opt_state))
        # Use -grads to maximize the score.
        opt_state = opt_update(stepnum, -grads, opt_state)
        return score, opt_state

    for stepnum in range(nsteps):
        params = opt_params(opt_state)
        score, opt_state = take_step(stepnum, opt_state)
        callback(np.float32(score), params)


def scipy_optimize(minimize_kwargs, initial_params, transform_fun, mixing_matrix, init_data,
                   gals_per_arcmin2, fsky, metric, callback, scale=-1.):
    """Optimize weights wrt parameterized dndz weights using scipy.

    Should normally be used via the driver :func:`optimize`.

    Uses a fast calculation of the metrics governed by init_data.

    Optimizes using :func:`scipy.optimize.minimize` initialized with the specified
    keyword args.
    """
    pshape = initial_params.shape

    @jax.jit
    def score(p):
        return metrics(p, transform_fun, mixing_matrix, init_data,  gals_per_arcmin2, fsky)[metric]

    score_and_grads = jax.jit(jax.value_and_grad(score))

    def optimize_fun(p):
        score, grads = score_and_grads(jnp.array(p.reshape(pshape)))
        callback(score, p)
        return scale * scores[-1], scale * np.array(grads).reshape(-1)

    result = scipy.optimize.minimize(optimize_fun, initial_params.flatten(), jac=True, **minimize_kwargs)
    if result.status not in (0, 2):
        print(result.message)


def optimize(nbin, mixing_matrix, init_data, ntrial=1, transform='softmax', method='jax', opt_args={},
             gals_per_arcmin2=20., fsky=0.25, metric='FOM_DETF_3x2', interval=500, seed=123):
    """Optimize a single 3x2 metric with respect to normalized dn/dz weights calculated from
    unconstrained parameters p as:

       1/n dn/dz = transform(p) . mixing_matrix

    Returns the best score and corresponding normalized dn/dz weights.
    """
    # Normalize the input mixing matrix.
    mixing_matrix = jnp.array(mixing_matrix / mixing_matrix.sum())
    nzopt, nzcalc = mixing_matrix.shape

    if transform == 'softmax':
        transform_fun = softmax_transform
        pshape = (nbin, nzopt)
    elif transform == 'extend':
        transform_fun = extend_transform
        pshape = (nbin - 1, nzopt)
    else:
        raise ValueError(f'Invalid transform: {transform}.')

    all_scores = []
    max_score = -1
    best_params = None
    def callback(score, params):
        nonlocal all_scores, max_score, best_params
        if score > max_score:
            max_score = score
            best_params = params
        all_scores[-1].append(score)
        if len(all_scores[-1]) % interval == 0:
            print(f'  score={score:.3f} (max={max_score:.3f}) after {len(all_scores[-1])} steps.')

    opt_args.update(
        transform_fun=transform_fun, mixing_matrix=mixing_matrix, init_data=init_data,
        gals_per_arcmin2=gals_per_arcmin2, fsky=fsky, metric=metric, callback=callback)
    if method == 'jax':
        optimizer = functools.partial(jax_optimize, **opt_args)
    elif method == 'scipy':
        optimizer = functools.partial(scipy_optimize, **opt_args)
    else:
        raise ValueError(f'Invalid method: {method}.')

    gen = np.random.RandomState(seed)
    max_score = -1
    for trial in range(ntrial):
        all_scores.append([])
        params_in = gen.normal(size=pshape)
        optimizer(initial_params=params_in)
        print(f'trial {trial+1}/{ntrial}: score={all_scores[-1][-1]:.3f} (max={max_score:.3f}) after {len(all_scores[-1])} steps.')

    # Calculate all 3x2 scores at the best parameters.
    best_scores = metrics(best_params, transform_fun, mixing_matrix, init_data, gals_per_arcmin2, fsky)
    best_scores = {metric: float(value) for metric, value in best_scores.items()}

    # Convert the best parameters to normalized weights and per-bin dn/dz.
    weights = transform_fun(best_params)
    dndz_bin = weights.dot(mixing_matrix)

    return best_scores, np.array(weights), np.array(dndz_bin), all_scores


def plot_dndz(dndz, zedges, gals_per_arcmin2=20.):
    """Plot the normalized dndz from an optimization.

    Should normally use zedges=init_data[0].
    """
    zc = 0.5 * (zedges[1:] + zedges[:-1])
    dz = np.diff(zedges)
    norm = gals_per_arcmin2 * 0.1 / dz
    for i, dndz_bin in enumerate(dndz):
        color = f'C{i}'
        plt.hist(zc, zedges, weights=dndz_bin * norm, histtype='stepfilled', alpha=0.25, color=color)
        plt.hist(zc, zedges, weights=dndz_bin * norm, histtype='step', alpha=0.75, color=color)
    plt.hist(zc, zedges, weights=dndz.sum(axis=0) * norm, histtype='step', color='k')
    plt.xlabel('Redshift $z$')
    plt.xlim(zedges[0], zedges[-1])
    plt.ylabel('Galaxies / ($\Delta z=0.1$) / sq.arcmin.')
