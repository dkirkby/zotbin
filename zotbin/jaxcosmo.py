"""Modifications to jax-cosmo
"""
import jax
import jax.numpy as jnp

import jax_cosmo.background
import jax_cosmo.power
import jax_cosmo.transfer
import jax_cosmo.utils
import jax_cosmo.constants
import jax_cosmo.probes


@jax.tree_util.register_pytree_node_class
class bin_nz(jc.redshift.redshift_distribution):
    """Defines a bin of constant dn/dz
    """
    def __init__(self, zlo, zhi, gals_per_arcmin2, zmax):
        super(bin_nz, self).__init__(
            zlo, zhi, # autograd variables
            gals_per_arcmin2=gals_per_arcmin2, zmax=zmax # base-class config
        )
        # Do no perform numerical normalization.
        self._norm = 1.

    def pz_fn(self, z):
        raise NotImplementedError

    def __call__(self, z):
        zlo, zhi = self.params
        return jnp.maximum(0., jnp.sign((z - zlo) * (zhi - z)) / (zhi - zlo))


def one_density_kernel(self, cosmo, z, ell, s=slice(None)):
    z = jnp.atleast_1d(z)
    # Extract parameters
    pzs, bias = self.params
    # Retrieve density kernel
    kernel = jax_cosmo.probes.density_kernel(cosmo, pzs[s], bias, z, ell)
    return kernel

jax_cosmo.probes.NumberCounts.one_kernel = one_density_kernel


def one_wl_kernel(self, cosmo, z, ell, s=slice(None)):
    z = jnp.atleast_1d(z)
    # Extract parameters
    pzs, m = self.params[:2]
    kernel = jax_cosmo.probes.weak_lensing_kernel(cosmo, pzs[s], z, ell)
    # If IA is enabled, we add the IA kernel
    if self.config["ia_enabled"]:
        bias = self.params[2]
        kernel += jax_cosmo.probes.nla_kernel(cosmo, pzs[s], bias, z, ell)
    # Applies measurement systematics
    if isinstance(m, list):
        m = jnp.expand_dims(np.stack([mi for mi in m], axis=0), 1)
    kernel *= 1.0 + m
    return kernel

jax_cosmo.probes.WeakLensing.one_kernel = one_wl_kernel


def new_angular_cl(
    cosmo, ell, probes, nagrid=512, transfer_fn=jax_cosmo.transfer.Eisenstein_Hu, nonlinear_fn=jax_cosmo.power.halofit
):
    """
    Computes angular Cls for the provided probes

    All using the Limber approximation

    Returns
    -------

    cls: [ell, ncls]
    """
    # Retrieve the maximum redshift probed
    zmax = max([p.zmax for p in probes])

    # Tabulate the scale-factor grid to use for integrations.
    amin, amax = jax_cosmo.utils.z2a(zmax), 1.
    a = jnp.linspace(amin, amax, nagrid + 1)
    da = (amax - amin) / nagrid

    # Precompute Simpson's rule quadrature coefficients as (1,2,4) * da / 3
    assert nagrid % 2 == 0
    idx = jnp.arange(nagrid + 1.)
    coef = (2 * (1 + (idx % 2)) - (idx % nagrid == 0)) * da / 3

    # Calculate the corresponding redshift and comoving distance grids.
    z = jax_cosmo.utils.a2z(a)
    chi = jax_cosmo.background.radial_comoving_distance(cosmo, a)

    # Calculate P(k, a).
    k = (ell.reshape(-1, 1) + 0.5) / jnp.clip(chi, 1.0)
    pk = jax_cosmo.power.nonlinear_matter_power(cosmo, k, a, transfer_fn, nonlinear_fn)

    # Add the remaining probe independent factors.
    f = pk * coef * jax_cosmo.background.dchioverda(cosmo, a) / jnp.clip(chi ** 2, 1.0) / jax_cosmo.constants.c ** 2

    results = []
    nprobe = len(probes)
    idx = 0
    k1 = 0
    for i1 in range(nprobe):
        n1 = probes[i1].n_tracers
        for j1 in range(n1):
            kernel1 = probes[i1].one_kernel(cosmo, z, ell.reshape(-1, 1), slice(j1, j1+1))
            k2 = k1
            for i2 in range(i1, nprobe):
                n2 = probes[i2].n_tracers
                j2min = j1 if i1 == i2 else 0
                for j2 in range(j2min, n2):
                    kernel2 = probes[i2].one_kernel(cosmo, z, ell.reshape(-1, 1), slice(j2, j2+1))
                    results.append(jnp.sum(f * kernel1 * kernel2, axis=1))
                    k2 += 1
                    idx += 1
            k1 += 1

    return jnp.vstack(results)
