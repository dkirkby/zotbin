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


def jax_optimize(optimizer, initial_params, nsteps, transform_fun, mixing_matrix, init_data,
                 gals_per_arcmin2=20., fsky=0.25, metric='FOM_DETF_3x2'):
    """Optimize weights wrt parameterized dndz weights using jax.

    Should normally be used via the driver :func:`optimize`.

    Uses a fast calculation of the metrics governed by init_data.

    Runs exactly nsteps and returns the max_score obtained and the corresponding best
    parameters and score history as numpy arrays.  Note that the best score is not
    necessarily from the last step.

    The input optimizer can any instantiated method from jax.experimental.optimizers,
    e.g. ``optimizers.sgd(learning_rate)``.
    """
    assert mixing_matrix.sum() == 1

    @jax.jit
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

    scores = []
    max_score = -1.
    final_params = initial_params
    for stepnum in range(nsteps):
        score, opt_state = take_step(stepnum, opt_state)
        scores.append(float(score))
        if scores[-1] > max_score:
            max_score = scores[-1]
            final_params = opt_params(opt_state)

    return max_score, np.array(final_params), np.array(scores)


def scipy_optimize(minimize_kwargs, initial_params, transform_fun, mixing_matrix, init_data,
                   scale=-1., gals_per_arcmin2=20., fsky=0.25, metric='FOM_DETF_3x2'):
    """Optimize weights wrt parameterized dndz weights using scipy.

    Should normally be used via the driver :func:`optimize`.

    Uses a fast calculation of the metrics governed by init_data.

    Optimizes using :func:`scipy.optimize.minimize` initialized with the specified
    keyword args.
    """
    assert mixing_matrix.sum() == 1
    pshape = initial_params.shape

    @jax.jit
    def score(p):
        return metrics(p, transform_fun, mixing_matrix, init_data,  gals_per_arcmin2, fsky)[metric]

    score_and_grads = jax.jit(jax.value_and_grad(score))

    scores = []
    max_score = -1
    final_params = initial_params

    def optimize_fun(p):
        nonlocal max_score, final_params
        score, grads = score_and_grads(jnp.array(p.reshape(pshape)))
        scores.append(np.float(score))
        if scores[-1] > max_score:
            max_score = scores[-1]
            final_params = p.reshape(pshape)
        return scale * scores[-1], scale * np.array(grads).reshape(-1)

    result = scipy.optimize.minimize(optimize_fun, initial_params.flatten(), jac=True, **minimize_kwargs)
    if result.status not in (0, 2):
        print(result.message)

    return max_score, final_params, np.array(scores)


def optimize(nbin, mixing_matrix, trial_fun, ntrial=1, transform='softmax', metric='FOM_DETF_3x2', seed=123, plot=True):
    """Optimize a single 3x2 metric with respect to normalized dn/dz weights calculated from
    unconstrained parameters p as:

       1/n dn/dz = transform(p) . mixing_matrix

    Returns the best score and corresponding parameter values.
    """
    if mixing_matrix.sum() != 1:
        raise ValueError('The mixing matrix is not normalized.')
    nzopt, nzcalc = mixing_matrix.shape

    if transform == 'softmax':
        transform_fun = softmax_transform
        pshape = (nbin, nzopt)
    elif transform == 'extend':
        transform_fun = extend_transform
        pshape = (nbin - 1, nzopt)

    gen = np.random.RandomState(seed)
    max_score = -1
    for trial in range(ntrial):
        params_in = gen.normal(size=pshape)
        score, params_out, scores = trial_fun(
            initial_params=params_in, transform_fun=transform_fun, mixing_matrix=mixing_matrix, metric=metric)
        print(f'trial {trial+1}/{ntrial}: score={score:.3f} (max={max_score:.3f}) after {len(scores)} steps.')
        if score > max_score:
            max_score = score
            best_params = params_out
        if plot:
            plt.plot(scores)
    if plot:
        plt.xlabel('Optimize steps')
        plt.ylabel(metric)

    return max_score, best_params
