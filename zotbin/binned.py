import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

import jax_cosmo.bias
import jax_cosmo.probes
import jax_cosmo.core

import zotbin.jaxcosmo
import zotbin.reweight


def get_zedges(z, frac=0.02, damin=0.025, damax=0.028, plot=False):
    zsorted = np.sort(z)
    asorted = 1 / (zsorted + 1)
    nfrac = int(np.round(frac * len(z)))
    hi = 0
    zedges = [zsorted[0]]
    last = len(z) - 1
    while hi < last:
        lo = hi
        hi = min(lo + nfrac, last)
        while (hi < last) and (asorted[hi] >= asorted[lo] - damin):
            hi += 1
        while (hi > lo + 1) and (asorted[hi] <= asorted[lo] - damax):
            hi -= 1
        zedges.append(zsorted[hi])
    if (asorted[lo] < asorted[hi] + damin) or (hi - lo + 1 < nfrac):
        # Merge last bin.
        zedges[-2] = zedges[-1]
        zedges = zedges[:-1]
    zedges = np.array(zedges)
    print(f'Selected {len(zedges)} edges.')
    aedges = 1 / (1 + zedges[::-1])
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].hist(z, range=(zsorted[0], zsorted[-1]), bins=500, alpha=0.5)
        ax[1].hist(1 / (1 + z), range=(asorted[-1], asorted[0]), bins=500, alpha=0.5)
        for ze in zedges:
            ax[0].axvline(ze, color='k', lw=1)
        for ae in aedges:
            ax[1].axvline(ae, color='k', lw=1)
    return zedges


def get_bin_probes(zedges, what='3x2', sigma_e=0.26, linear_bias=1., gals_per_arcmin2=1.):

    nzbins = len(zedges) - 1
    zbins = []
    for i in range(nzbins):
        zbin = zotbin.jaxcosmo.bin_nz(
            zedges[i], zedges[i + 1],
            gals_per_arcmin2=gals_per_arcmin2, zmax=zedges[-1])
        zbins.append(zbin)

    probes = []
    # start with number counts
    if (what == 'gg' or what == '3x2'):
        # Define a bias parameterization
        bias = jax_cosmo.bias.inverse_growth_linear_bias(linear_bias)
        probes.append(jax_cosmo.probes.NumberCounts(zbins, bias))

    if (what == 'ww' or what == '3x2'):
        probes.append(jax_cosmo.probes.WeakLensing(zbins, sigma_e=sigma_e))

    return probes


def init_binned_cl(zedges, ell, Omega_c=0.27, sigma8=0.8404844953840714, w0=-1., wa=0., what='3x2', sigma_e=0.26, linear_bias=1., nagrid=1024):
    """
    """
    probes = get_bin_probes(zedges, what, sigma_e, linear_bias, gals_per_arcmin2=1.)
    def get_cl(p0, p1, p2, p3):
        model = jax_cosmo.core.Cosmology(
            Omega_c = p0,
            Omega_b = 0.045,
            h = 0.67,
            n_s = 0.96,
            sigma8 = p1,
            Omega_k=0.,
            w0=p2,
            wa=p3)
        return zotbin.jaxcosmo.new_angular_cl(model, ell, probes, nagrid)
    cl = []
    for i in range(4):
        get_cl_grad = jax.jacfwd(get_cl, i)
        cl.append(get_cl_grad(Omega_c, sigma8, w0, wa))
        print(i, cl[-1].shape)
    cl.append(get_cl(Omega_c, sigma8, w0, wa))
    print(cl[-1].shape)
    return zotbin.reweight.init_reweighting(probes, cl)


def save_binned(name, zedges, ell, ngals, noise, cl_in):
    keys = dict(zedges=zedges, ell=ell, ngals=ngals, noise=noise)
    nprobe = len(ngals)
    for i in range(nprobe):
        for j in range(i + 1):
            keys[f'cl_{i}_{j}'] = cl_in[i][j]
    np.savez(name, **keys)


def load_binned(name):
    with np.load(name) as keys:
        zedges = jnp.array(keys['zedges'])
        ell = jnp.array(keys['ell'])
        ngals = jnp.array(keys['ngals'])
        noise = jnp.array(keys['noise'])
        nprobe = len(ngals)
        cl_in = []
        for i in range(nprobe):
            cl_in.append([])
            for j in range(i + 1):
                cl_in[-1].append(jnp.array(keys[f'cl_{i}_{j}']))
    return zedges, ell, ngals, noise, cl_in


def get_binned_weights(zedges, z, idx):
    """Calculate the weight matrix for galaxies with redshifts z and corresponding output bins idx.

    Any galaxies with z < zedges[0] or z >= zedges[-1] contribute to the normalization only.
    """
    idx_out = np.array(idx, int)
    assert len(z) == len(idx_out)
    ntot = len(z)
    nout = idx_out.max() + 1
    nin = len(zedges) - 1
    idx_in = np.digitize(z, zedges) - 1
    inbounds = (idx_in >= 0) & (idx_in < nin)
    weights = []
    for ibin in range(nout):
        weights.append(np.bincount(idx_in[(idx_out == ibin) & inbounds], minlength=nin))
    return np.vstack(weights) / ntot


def get_binned_scores(idx, z, zedges, ell, ngals, noise, cl_in, gals_per_arcmin2=20., fsky=0.25):
    """
    """
    weights = get_binned_weights(zedges, z, idx)
    return zotbin.reweight.reweighted_metrics(
        weights, ell, ngals, noise, cl_in, gals_per_arcmin2, fsky)
