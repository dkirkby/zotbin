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


# This is currently several times slower than reweight_cl so needs more tuning.
@functools.partial(jax.jit, static_argnums=(1, 4))
def reweight_noise_cl(weights, gals_per_arcmin2, ngals, noise, nell):
    """
    """
    #assert len(weights) == len(noise)
    nprobe = weights.shape[0]
    noise_out = []
    ntracers = 0
    for i in range(nprobe):
        noise_inv_in = 1 / (ngals[i] * noise[i])
        noise_inv_out = gals_per_arcmin2 * weights[i].dot(noise_inv_in)
        noise_out.append(1 / noise_inv_out)
        ntracers += len(noise_inv_out)
    noise = jnp.concatenate(noise_out)

    # Define an ordering for the blocks of the signal vector
    cl_index = []
    for i in range(ntracers):
        for j in range(i, ntracers):
            cl_index.append((i, j))
    cl_index = jnp.array(cl_index)

    # Only include a noise contribution for the auto-spectra
    def get_noise_cl(inds):
        i, j = inds
        delta = 1.0 - jnp.clip(jnp.abs(i - j), 0.0, 1.0)
        return noise[i] * delta * jnp.ones(nell)

    return jax.lax.map(get_noise_cl, cl_index), cl_index


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


def reweighted_cov(cl_out, nl_out, cl_index, ell, fsky=0.25):
    """
    """
    # This is essentially jc.angular_cl.gaussian_cl_covariance without using probes...
    cl_obs = cl_out + nl_out
    ncl = cl_obs.shape[0]
    norm = (2 * ell + 1) * jnp.gradient(ell) * fsky

    def find_index(a, b):
        if (a, b) in cl_index:
            return cl_index.index((a, b))
        else:
            return cl_index.index((b, a))

    cov_blocks = []
    for (i, j) in cl_index:
        for (m, n) in cl_index:
            cov_blocks.append(
                (find_index(i, m), find_index(j, n), find_index(i, n), find_index(j, m))
            )

    def get_cov_block(inds):
        a, b, c, d = inds
        cov = (cl_obs[a] * cl_obs[b] + cl_obs[c] * cl_obs[d]) / norm
        return cov

    # Build a sparse representation of the output covariance.
    return jax.lax.map(get_cov_block, jnp.array(cov_blocks)).reshape((ncl, ncl, len(ell)))


def reweighted_metrics(weights, ell, ngals, noise, cl_in, gals_per_arcmin2, fsky):
    """
    """
    nell = len(ell)
    cl_out = reweight_cl(weights, ngals, cl_in)
    nl_out, cl_index = reweight_noise_cl(weights, gals_per_arcmin2, ngals, noise, nell)
    cov_out = reweighted_cov(cl_out[-1], nl_out, cl_index, ell, fsky)

    # Calculate SNR2 = mu^t . Cinv . mu
    cinv = sparse.inv(cov_out)
    mu = cl_out.reshape(-1, 1)
    snr = jnp.sqrt(sparse.dot(mu.T, cinv, mu)[0, 0])

    # Calculate FoM

    return {'SNR_3x2': snr}
