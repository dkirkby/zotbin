import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

import jax_cosmo.bias
import jax_cosmo.probes
import jax_cosmo.core
import jax_cosmo.parameters
import jax_cosmo.background

import zotbin.jaxcosmo
import zotbin.reweight


def get_zedges(zmax, nzbin, zplot=None):
    """Calculate redshift bin edges equally spaced in comoving distance.
    """
    # Tabulate comoving distance over a grid spanning the full range of input redshifts.
    zgrid = np.linspace(0, zmax, 1000)
    agrid = 1 / (1 + zgrid)
    model = jax_cosmo.parameters.Planck15()
    chi_grid = jax_cosmo.background.radial_comoving_distance(model, agrid)
    # Compute bin edges that are equally spaced in chi.
    chi_edges = np.linspace(0, chi_grid[-1], nzbin + 1)
    zedges = np.empty(nzbin + 1)
    zedges[0] = 0.
    zedges[-1] = zmax
    zedges[1:-1] = np.interp(chi_edges[1:-1], chi_grid, zgrid)
    aedges = 1 / (1 + zedges)
    if zplot is not None:
        plots = {
            'Redshift $z$': zplot,
            'Scale factor $a$': 1 / (1 + zplot),
            'Comoving distance $\chi$ [Mpc]': np.interp(zplot, zgrid, chi_grid)
        }
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, label in zip(axes, plots):
            ax.hist(plots[label], bins=2 * nzbin, histtype='stepfilled', color='r', alpha=0.5)
            ax.set_xlabel(label)
        for ax, edges in zip(axes, (zedges, aedges, chi_edges)):
            ax.set_yticks([])
            ax.set_xlim(edges[0], edges[-1])
            for edge in edges[1:-1]:
                ax.axvline(edge, c='k', lw=1, alpha=0.5)
        plt.tight_layout()
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


def init_binned_cl(zedges, ell, what='3x2', sigma_e=0.26, linear_bias=1., nagrid=1024):
    """
    """
    probes = get_bin_probes(zedges, what, sigma_e, linear_bias, gals_per_arcmin2=1.)
    def get_cl(p0, p1, p2, p3, p4, p5, p6):
        model = jax_cosmo.core.Cosmology(
            Omega_c = p0,
            Omega_b = p1,
            h = p2,
            n_s = p3,
            sigma8 = p4,
            Omega_k=0.,
            w0=p5,
            wa=p6)
        return zotbin.jaxcosmo.new_angular_cl(model, ell, probes, nagrid)
    cl = []
    fiducial = [0.27, 0.045, 0.67, 0.96, 0.840484495, -1.0, 0.0]
    for i in range(7):
        get_cl_grad = jax.jacfwd(get_cl, i)
        cl.append(get_cl_grad(*fiducial))
        print(i, cl[-1].shape)
    cl.append(get_cl(*fiducial))
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
    """Calculate all of the 3x2 metrics: SNR2, FOM, FOM_DETF.
    """
    w = get_binned_weights(zedges, z, idx)
    weights = jnp.array([w, w])
    return zotbin.reweight.reweighted_metrics(
        weights, ell, ngals, noise, cl_in, gals_per_arcmin2, fsky)
