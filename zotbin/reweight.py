import functools

import numpy as np

import jax
import jax.numpy as jnp

import jax_cosmo.sparse as sparse


def init_reweighting(probes, cl):
    """
    """
    nprobe = len(probes)
    ngrad = len(cl)
    ncl, nell = cl[0].shape
    nzbin = np.array([probe.n_tracers for probe in probes])
    breaks = np.concatenate(([0], np.cumsum(nzbin)))
    nztot = breaks[-1]
    assert ncl == nztot * (nztot + 1) // 2
    ngals = [np.array([pz.gals_per_arcmin2 for pz in probe.params[0]], np.float32) for probe in probes]
    noise = [ np.array(probe.noise()) for probe in probes]
    cl_in = [[np.empty((ngrad, nzbin[i], nzbin[j], nell), np.float32) for j in range(i + 1)] for i in range(nprobe)]

    idx, j1 = 0, 0
    for i1 in range(nztot):
        if i1 == breaks[j1 + 1]:
            j1 += 1
        k1 = i1 - breaks[j1]
        j2 = j1
        for i2 in range(i1, nztot):
            if i2 == breaks[j2 + 1]:
                j2 += 1
            k2 = i2 - breaks[j2]
            for q in range(ngrad):
                cl_in[j2][j1][q, k1, k2] = cl[q][idx]
            if j1 == j2 and k1 != k2:
                assert k2 > k1
                # Symmetrize off-diagonal elements of the diagonal blocks.
                for q in range(ngrad):
                    cl_in[j2][j1][q, k2, k1] = cl_in[j2][j1][q, k1, k2]
            idx += 1

    return ngals, noise, cl_in


@jax.jit
def reweight_cl(weights, ngals, cl_in):
    """
    """
    # assert len(weights) == len(ngals)
    nprobe = weights.shape[0]
    offset = 0
    w = [None] * nprobe
    nzbin = np.array([len(W) for W in weights])
    nout = np.sum(nzbin * (1 + np.arange(nprobe)))
    cl_out = [None] * nout
    for i1 in range(nprobe):
        nrow = len(weights[i1])
        rowstep = nprobe - i1
        for i2 in range(i1, nprobe):
            #assert weights[i2].shape[1] == len(ngals[i2])
            W = weights[i2] * ngals[i2]
            W /= jnp.sum(W, axis=1, keepdims=True)
            w[i2] = W
            cl = jnp.einsum('ip,spqk,jq->sijk', w[i1], cl_in[i2][i1], w[i2])
            for j in range(nrow):
                start = j if i1 == i2 else 0
                cl_out[offset + j * rowstep + i2 - i1] = cl[:, j, start:]
        offset += nrow * rowstep
    return jnp.concatenate(cl_out, axis=1)


@jax.jit
def reweight_noise_cl(weights, ngals, noise, gals_per_arcmin2):
    """
    """
    noise_out = []
    for i in range(weights.shape[0]):
        noise_inv_in = 1 / (ngals[i] * noise[i])
        noise_inv_out = gals_per_arcmin2 * weights[i].dot(noise_inv_in)
        noise_out.append(1 / noise_inv_out)
    nonzero = jnp.concatenate(noise_out)
    ntracers = nonzero.shape[0]
    i = jnp.arange(ntracers)
    scatter = (2 * ntracers - i + 1) * i // 2
    ncl = (ntracers + 1) * ntracers // 2
    return jax.ops.index_update(
        jnp.zeros(ncl), scatter, nonzero, indices_are_sorted=True, unique_indices=True).reshape(ncl, 1)


_cov_indices_cache = {}

def get_cov_indices(ncl):

    if ncl not in _cov_indices_cache:
        i_of_k = np.empty(ncl, np.int32)
        j_of_k = np.empty(ncl, np.int32)
        k_of_ij = np.empty((ncl, ncl), np.int32)
        ntracer = int(np.round((np.sqrt(8 * ncl + 1) - 1) / 2))
        k = 0
        for i in range(ntracer):
            for j in range(i, ntracer):
                i_of_k[k] = i
                j_of_k[k] = j
                k_of_ij[j, i] = k_of_ij[i, j] = k
                k += 1
        p = k_of_ij[i_of_k.reshape(-1, 1), i_of_k]
        q = k_of_ij[j_of_k.reshape(-1, 1), j_of_k]
        r = k_of_ij[i_of_k.reshape(-1, 1), j_of_k]
        s = k_of_ij[j_of_k.reshape(-1, 1), i_of_k]
        _cov_indices_cache[ncl] = jnp.array((p, q, r, s))

    return _cov_indices_cache[ncl]


@jax.jit
def gaussian_cl_covariance_helper(ell, cl_signal, cl_noise, idx, f_sky):
    """Optimized kernel for assembling a sparse Gaussian covariance.

    Use :func:`get_cov_pq` to obtain (p, q) or use the wrapper
    :func:`gaussian_cl_covariance`.
    """
    cl_obs = cl_signal + cl_noise
    norm = (2 * ell + 1) * jnp.gradient(ell) * f_sky
    outer = cl_obs.reshape(-1, 1, len(ell)) * cl_obs / norm
    return outer[idx[0], idx[1]] + outer[idx[2], idx[3]]


def gaussian_cl_covariance(ell, cl_signal, cl_noise, f_sky=0.25, sparse=True):
    ell = jnp.atleast_1d(ell)
    idx = get_cov_indices(len(cl_signal))
    cov = gaussian_cl_covariance_helper(ell, cl_signal, cl_noise, idx, f_sky)
    return cov if sparse else sparse.to_dense(cov)


@jax.jit
def reweighted_metrics(weights, ell, ngals, noise, cl_in, gals_per_arcmin2, fsky):
    """The sparse cinv calculation is currently the bottleneck.
    """
    cl_out = reweight_cl(weights, ngals, cl_in)
    nl_out = reweight_noise_cl(weights, ngals, noise, gals_per_arcmin2)
    cov_out = gaussian_cl_covariance(ell, cl_out[-1], nl_out, fsky)
    cinv = sparse.inv(cov_out)
    mu = cl_out[-1].reshape(-1, 1)
    dmu = cl_out[:-1].reshape(7, -1)
    F = sparse.dot(dmu, cinv, dmu.T)
    Finv = jnp.linalg.inv(F)
    return {
        'SNR_3x2': jnp.sqrt(sparse.dot(mu.T, cinv, mu)[0, 0]),
        'FOM_3x2': 1 / (6.17 * jnp.pi * jnp.sqrt(jnp.linalg.det(Finv[jnp.ix_([0,4], [0,4])]))),
        'FOM_DETF_3x2': 1 / (6.17 * jnp.pi * jnp.sqrt(jnp.linalg.det(Finv[jnp.ix_([5,6], [5,6])])))
    }
