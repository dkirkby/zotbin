import numpy as np
import matplotlib.pyplot as plt


def get_fedges(features, npct):
    """Calculate percentile edges along each feature axis.
    """
    ndata, nfeat = features.shape
    # Calculate percentile bins for each feature.
    pct = np.linspace(0, 100, npct + 1)
    return np.percentile(features, pct, axis=0).T


def fdigitize(features, fedges):
    """Digitize samples into feature bins.
    """
    ndata, nfeat = features.shape
    npct = fedges.shape[1] - 1
    mul = 1
    D = features.T
    sample_bin = np.zeros(ndata, int)
    for i in range(nfeat):
        sample_bin += mul * np.clip(np.digitize(D[i], fedges[i]) - 1, 0, npct - 1)
        mul *= npct
    return sample_bin


def groupinit(features, redshift, zedges, npct=20):
    """Initialize the calculation of feature bin groups.
    """
    ndata, nfeat = features.shape
    assert redshift.shape == (ndata,)
    nzbin = len(zedges) - 1
    nfbin = int(npct ** nfeat)
    # Calculate feature bin edges and assign each sample to a bin.
    fedges = get_fedges(features, npct)
    sample_bin = fdigitize(features, fedges)
    # Histogram the redshift distribution in each feature bin.
    zhist = np.empty((nfbin, nzbin), int)
    for i in range(nfbin):
        zhist[i], _ = np.histogram(redshift[sample_bin == i], zedges)
    return fedges, sample_bin, zhist


def groupbins(features, redshift, zedges, npct, weighted=True, sigma=0.2,
              ngrp_save=(400, 300, 200, 150, 100, 50), maxfrac=0.02, minfrac=0.005,
              validate=False, validate_interval=1000, maxiter=None, savename='group_{N}.npz'):
    """Group similar bins in multidimensional feature space.

    Initial bins are defined as a rectangular grid in feature space with a grid
    defined such that the projections onto each feature axis contain an equal
    number of samples.  Any bins that contain no galaxies form one group and
    are removed from subsequent analysis.

    Similarity is defined as the product of independent feature and redshift similarities,
    with values in the range 0-1 and 1 indicating maximum similarity.

    Feature similarity is defined as exp(-(dr / sigma) ** 2) where dr is the Euclidean
    (grid index) separation in the multimensional feature space rectangular grid.
    The hyperparameter sigma controls the relative importance of the feature and
    redshift similarities in the subsequent grouping. Values of dr are normalized
    such that dr=1 corresonds to the full range of each feature, i.e., the grid
    size along each feature axis.

    Redshift similarity is based on the histogram of redshifts associated with each
    feature bin, interpreted as a vector.  When weighted is False, similarity is
    calculated as the cosine of the angle between two vectors.  Since all components
    of the redshift vector are histogram bin contents >= 0, the resulting cosine
    simililarities are all in the range 0-1

    Since cosine similarity uses normalized vectors, it does not give more weight to
    a feature bin containing more samples.  An alternative, when weighted is True,
    is to use a similarity score of |z1 + z2| / (|z1| + |z2|), which is equivalent
    to a weighted average of the cosine similarities between z1+z2 and z1 or z2,
    respectively, with weights wk = |zk| / (|z1| + |z2|).

    Feature bins are grouped iteratively, by combining the pair of groups with the
    maximum similiarity, until either a minimum number of groups is reached or else
    all remaining groups are above a minimum sample threshold.  There is also a
    maximum sample threshold, and pairs whose combination would exceed this
    threshold are never combined.

    For testing purposes, grouping can also be terminated after a fixed number
    of iterations.  The incremental updates of the feature and redshift similarity
    matrices can also be periodically validated against much slower calculations
    from scratch.

    When two groups are merged, their redshift histograms are added to calculate
    updated feature similarities with all other remaining groups.  The updated
    feature similarities use the minimum separation dr between the bins of
    the merged group and bins of each other group. This ensures that the maximum
    feature similarity occurs between adjacent groups, regardless of their size.
    """
    fedges, sample_bin, zhist = groupinit(features, redshift, zedges, npct)
    ndata, nfeat = features.shape
    nfbin, nzbin = zhist.shape
    nmax = int(np.round(maxfrac * ndata))
    nmin = int(np.round(minfrac * ndata))
    assert nmax > nmin
    nzbin = len(zedges) - 1
    nfbin = int(npct ** nfeat)
    print(f'Grouping with ndata={ndata}, nmin={nmin}, nmax={nmax}, nzbin={nzbin}, nfbin={nfbin}.')
    if validate:
        zhist0 = zhist.copy()
    else:
        zhist0 = None
    min_groups = min(ngrp_save)

    # Calculate the number of samples (1-norm) in each feature bin.
    zsum = np.sum(zhist, axis=1)

    # Calculate the Euclidean 2-norm of the redshift vector for each feature bin.
    znorm = np.sqrt(np.sum(zhist ** 2, axis=1))

    # Assign initial group ids.
    grpid = np.arange(nfbin, dtype=int)
    active = np.ones(nfbin, bool)
    if np.any(znorm == 0):
        # Group all empty bins and mark them inactive.
        empty = zsum == 0
        active[empty] = False
        iempty = np.min(grpid[empty])
        grpid[empty] = iempty
    else:
        iempty = None
    ngrp = np.count_nonzero(active)

    # Initialize masks to zero elements on the diagonal or also below.
    nondiag = 1 - np.identity(nfbin)
    upper = np.triu(np.ones((nfbin, nfbin)), 1)

    # Calculate the normalized Euclidean distance matrix between feature bins.
    x = np.arange(npct) / (npct - 1)
    dims = np.meshgrid(*([x] * nfeat), sparse=False, copy=False)
    r1 = np.stack(dims, axis=-1).reshape(-1, nfeat)
    r2 = r1.reshape(-1, 1,  nfeat)
    dr2 = np.sum((r1 - r2) ** 2, axis=-1) / sigma ** 2
    assert dr2.shape == (nfbin, nfbin)

    # Calculate the initial feature similarity matrix.
    # Bins that are sufficiently far apart (relative to sigma) will
    # have fsim == 0, due to numerical precision, and will therefore
    # never be grouped.
    fsim = np.exp(-dr2)

    # Calculate the initial redshift similarity matrix.
    if weighted:
        pairs = zhist.reshape(nfbin, 1, nzbin) + zhist
        pairnorm = np.sqrt(np.sum(pairs ** 2, axis=-1))
        sumnorm = znorm.reshape(nfbin, 1) + znorm
        zsim = np.divide(pairnorm, sumnorm, where=sumnorm > 0, out=np.zeros_like(sumnorm))
    else:
        unit = np.divide(zhist, znorm.reshape(-1, 1), where=znorm.reshape(-1, 1) > 0, out=np.zeros_like(zhist, float))
        zsim = np.einsum('ij,kj->ik', unit, unit)

    if np.any(~active):
        inactive = np.where(~active)[0]
        fsim[inactive, :] = fsim[:, inactive] = 0.
        zsim[inactive, :] = zsim[:, inactive] = 0.
    fsim *= nondiag
    zsim *= nondiag

    # Initialize a mask of pairs that should not be grouped because that would exceed maxfrac.
    # Only elements above the diagonal will be used.
    eligible = np.ones((nfbin, nfbin))

    # Run iterative grouping.
    niter = 0
    while (ngrp > min_groups) and (np.min(zsum[active]) < nmin):
        # Find the pair of bins that are most similar.
        similarity = fsim * zsim * upper * eligible
        i1, i2 = np.unravel_index(np.argmax(similarity), similarity.shape)
        ##print(ngrp, i1, i2, zsim[i1, i2], fsim[i1, i2])
        assert i1 < i2
        assert active[i1] and active[i2]
        assert grpid[i1] == i1 and grpid[i2] == i2
        assert np.sum(zhist[active]) == ndata
        # Are we still below the threshold if we combine i1 and i2?
        if zsum[i1] + zsum[i2] <= nmax:
            # Merge i1 and i2 into i1.
            active[i2] = False
            sel1 = (grpid == i1)
            sel2 = (grpid == i2)
            ##print(np.where(sel1)[0], np.where(sel2)[0])
            grpid[sel1] = i1
            grpid[sel2] = i1
            # Update array values associated with i1.
            zhist[i1] += zhist[i2]
            oldi1znorm = znorm[i1]
            znorm[i1] = np.sqrt(np.sum(zhist[i1] ** 2))
            zsum[i1] += zsum[i2]
            # Calculate the redshift similarity of the merged (i1,i2) with all other bins.
            if weighted:
                pairs[i1, :] += zhist[i2]
                pairs[:, i1] += zhist[i2]
                i1pairnorm = np.sqrt(np.sum(pairs[i1] ** 2, axis=1))
                i1sumnorm = znorm + znorm[i1]
                newzsim = i1pairnorm / i1sumnorm
            else:
                unit[i1] = zhist[i1] / znorm[i1]
                newzsim = unit.dot(unit[i1])
            newzsim[i1] = 0.
            newzsim[~active] = 0.
            # Update the full redshift similarity matrix.
            zsim[i1, :] = zsim[:, i1] = newzsim
            zsim[i2, :] = zsim[:, i2] = 0.
            # Calculate the feature similarity of the merged (i1,i2) with all other bins.
            newfsim = np.maximum(fsim[i1], fsim[i2])
            newfsim[i1] = 0.
            newfsim[~active] = 0.
            # Update the full feature similarity matrix.
            fsim[i1, :] = fsim[:, i1] = newfsim
            fsim[i2, :] = fsim[:, i2] = 0.
            # Reduce the number of groups and save a snapshot if requested.
            ngrp -= 1
            if ngrp in ngrp_save:
                print(f'Finalizing result with {ngrp} groups...')
                grpid_save, zhist_save, zsim_save = finalize(
                    ngrp, grpid, zhist, zsim, zedges, active, sample_bin, redshift, iempty, validate, zhist0)
                if validate:
                    print('Validating saved results...')
                    validate_groups(features, redshift, zedges, fedges, grpid_save, zhist_save)
                fname = savename.format(N=ngrp)
                save_groups(fname, zedges, fedges, grpid_save, zhist_save, zsim_save)
                print(f'Saved {ngrp} groups to {fname}')
        else:
            # Zero this similiarity but leave i & j eligible for grouping with other bins.
            eligible[i1, i2] = 0.
        niter += 1
        if validate and (niter % validate_interval == 0):
            print(f'validating similarity matrices after {niter} iterations')
            zsim0 = redshift_similarity(zhist, znorm, grpid, active, weighted)
            assert validated(zsim, zsim0)
            fsim0 = feature_similarity(dr2, grpid, active)
            assert validated(fsim, fsim0)
        if niter == maxiter:
            print(f'Reached maxiter={maxiter}.')
            break
        if niter % 100 == 0:
            print(f'Reduced to {ngrp} groups after {niter} iterations.')
    if ngrp == min_groups:
        print(f'Reached min_groups={min_groups} after {niter} iterations.')
    elif np.min(zsum[active]) >= nmin:
        print(f'All groups above nmin={nmin} after {niter} iterations.')


def finalize(ngrp, grpid_in, zhist_in, zsim_in, zedges, active, sample_bin, redshift, iempty, validate, zhist0):
    """
    """
    ndata = len(sample_bin)
    nfbin, nzbin = zhist_in.shape
    grpid = grpid_in.copy()
    # Renumber the groups consecutively and compress the outputs.
    sample_grp = np.empty_like(sample_bin)
    if iempty != None:
        empty_bins = np.where(grpid == iempty)[0]
        sample_grp[np.isin(sample_bin, empty_bins)] = -1
        grpid[empty_bins] = -1
    ids = np.unique(grpid)
    if iempty != None:
        ids = ids[1:]
    assert ngrp == len(ids), f'ngrp mismatch: {ngrp} != {len(ids)}.'
    if validate:
        zhist_check = np.empty((ngrp, nzbin), int)
    rows = np.empty(ngrp, int)
    for i, idx in enumerate(ids):
        sel = (grpid == idx)
        bins = np.where(sel)[0]
        if validate:
            zhist_check[i] = zhist0[bins].sum(axis=0)
        assert bins[0] == idx
        assert active[bins[0]] and not np.any(active[bins[1:]])
        rows[i] = idx
        grp_bins = np.where(grpid == idx)[0]
        bin_sel = np.isin(sample_bin, grp_bins)
        assert bin_sel.shape == (ndata,)
        sample_grp[bin_sel] = i
        grp_hist, _ = np.histogram(redshift[bin_sel], zedges)
        assert np.array_equal(grp_hist, zhist_in[idx])
        grpid[bins] = i

    # Sort the groups in order of increase average redshift.
    zavg = np.sum(zhist_in[rows] * np.arange(nzbin), axis=1) / np.sum(zhist_in[rows], axis=1)
    iorder = np.argsort(zavg)
    rows = rows[iorder]
    grpid_out = np.empty_like(grpid)
    for i in range(ngrp):
        grpid_out[grpid == iorder[i]] = i
    if iempty is not None:
        grpid_out[grpid == -1] = -1
    zhist_out = zhist_in[rows]
    assert zhist_out.sum() == ndata
    if validate:
        assert np.array_equal(zhist_out, zhist_check[iorder])
    zsim_out = zsim_in[np.ix_(rows, rows)]
    assert np.all(np.diag(zsim_out) == 0)
    return grpid_out, zhist_out, zsim_out


def save_groups(fname, zedges, fedges, grpid, zhist, zsim):
    np.savez(fname, zedges=zedges, fedges=fedges, grpid=grpid, zhist=zhist, zsim=zsim)


def load_groups(fname):
    with np.load(fname) as keys:
        zedges = keys['zedges']
        fedges = keys['fedges']
        grpid = keys['grpid']
        zhist = keys['zhist']
        zsim = keys['zsim']
    return zedges, fedges, grpid, zhist, zsim


def pairs_iterator(grpid, active):
    """Iterate over pairs of groups and perform some validations.
    """
    ids = np.unique(grpid)
    for i, id1 in enumerate(ids):
        sel1 = grpid == id1
        idx1 = np.where(sel1)[0]
        assert np.all(grpid[idx1] == np.min(idx1))
        if not np.any(active[sel1]):
            continue
        # Only the first bin in each group must be active.
        assert active[idx1[0]] and not np.any(active[idx1[1:]])
        for id2 in ids[i + 1:]:
            sel2 = grpid == id2
            idx2 = np.where(sel2)[0]
            assert np.all(grpid[idx2] == np.min(idx2))
            if not np.any(active[sel2]):
                continue
            # Only the first bin in each group must be active.
            assert active[idx2[0]] and not np.any(active[idx2[1:]])
            # Groups must be disjoint.
            assert not np.any(sel1 & sel2)
            yield idx1, idx2


def feature_similarity(dr2, grpid, active):
    """Calculate the feature similarity matrix from scratch.
    This is relatively slow so only used to validate faster incremental updates.
    """
    fsim = np.zeros_like(dr2)
    for idx1, idx2 in pairs_iterator(grpid, active):
        # Find the minimum separation between bins in these groups.
        dr2sub = dr2[np.ix_(idx1, idx2)]
        dr2min = np.min(dr2sub)
        # Save the result.
        gnorm = np.exp(-dr2min)
        fsim[idx1[0], idx2[0]] = gnorm
        fsim[idx2[0], idx1[0]] = gnorm
    # The similarities of inactive rows and columns should already be zero.
    assert np.all(fsim[~active, :] == 0)
    assert np.all(fsim[:, ~active] == 0)
    return fsim


def redshift_similarity(zhist, znorm, grpid, active, weighted):
    """Calculate the redshift similarity matrix from scratch.
    This is relatively slow so only used to validate faster incremental updates.
    """
    nfbin, nzbin = zhist.shape
    zsim = np.zeros((nfbin, nfbin))
    for idx1, idx2 in pairs_iterator(grpid, active):
        idx1 = idx1[0]
        idx2 = idx2[0]
        assert znorm[idx1] > 0 and znorm[idx2] > 0
        if weighted:
            z12norm = np.sqrt(np.sum((zhist[idx1] + zhist[idx2]) ** 2))
            z12sim = z12norm / (znorm[idx1] + znorm[idx2])
        else:
            z12dot = np.sum(zhist[idx1] * zhist[idx2])
            z12sim = z12dot / (znorm[idx1] * znorm[idx2])
        zsim[idx1, idx2] = z12sim
        zsim[idx2, idx1] = z12sim
    # The similarities of inactive rows and columns should already be zero.
    assert np.all(zsim[~active, :] == 0)
    assert np.all(zsim[:, ~active] == 0)
    return zsim


def validated(M1, M2):
    """Test if M1 and M2 are the same matrix"""
    assert M1.shape == M2.shape
    if np.allclose(M1, M2):
        return True
    else:
        diff = np.abs(M1 - M2)
        i, j = np.unravel_index(np.argmax(diff), M1.shape)
        print(f'max diff at [{i},{j}]: |{M1[i,j]}-{M2[i,j]}| = {diff[i,j]}')
        return False


def validate_groups(features, redshift, zedges, fedges, grpid, zhist):
    """Validate the final results from groupbins()
    """
    npct = fedges.shape[1] - 1
    fedges, sample_bin, zhist0 = groupinit(features, redshift, zedges, npct)
    ngrp = grpid.max()
    for idx in range(ngrp):
        grp_bins = np.where(grpid == idx)[0]
        bin_sel = np.isin(sample_bin, grp_bins)
        zhist_test1, _ = np.histogram(redshift[bin_sel], zedges)
        zhist_test2 = zhist0[grp_bins].sum(axis=0)
        assert np.array_equal(zhist_test1, zhist[idx])
        assert np.array_equal(zhist_test2, zhist[idx])


def plotfbins(features, npct=20, nhist=100, inset_pct=1, show_edges=True):
    ndata, nfeat = features.shape
    edges = get_fedges(features, npct)
    D = features.T

    # Calculate inset ranges for histograms.
    ranges = np.percentile(D, (inset_pct, 100 - inset_pct), axis=1).T

    fig, axes = plt.subplots(nfeat, nfeat, figsize=(12, 12))
    for i in range(nfeat):
        ax = axes[i, i]
        ax.hist(D[i], nhist, range=ranges[i])
        ax.set_xlim(*ranges[i])
        ax.set_yticks([])
        if show_edges:
            for edge in edges[i]:
                ax.axvline(edge, c='r', alpha=0.5, lw=1)
        for j in range(i + 1, nfeat):
            ax = axes[j, i]
            hist, _, _ = np.histogram2d(D[i], D[j], [edges[i], edges[j]])
            ax.imshow(hist, origin='lower', interpolation='none', extent=[0, 100, 0, 100], vmin=0, vmax=hist.max())
            ax = axes[i, j]
            ax.hist2d(D[j], D[i], nhist, range=(ranges[j], ranges[i]))
            if show_edges:
                for edge in edges[j]:
                    ax.axvline(edge, c='r', alpha=0.5, lw=1)
                for edge in edges[i]:
                    ax.axhline(edge, c='r', alpha=0.5, lw=1)
    plt.tight_layout()


def plotzvecs(features, redshift, zedges, npct=20, pnorm=2, sort=True):

    _, _, zhist = groupinit(features, redshift, zedges, npct)
    nfbin, nzbin = zhist.shape

    zhist = zhist.astype(float)
    iz = np.arange(nzbin)
    zmean = np.sum(iz * zhist, axis=1) / np.sum(zhist, axis=1)
    order = np.argsort(zmean)
    znorm = np.sum(zhist ** pnorm, axis=1, keepdims=True) ** (1 / pnorm)
    zhist = np.divide(zhist, znorm, where=znorm > 0, out=zhist)

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.axis('off')
    ax.imshow(zhist[order], aspect='auto', origin='lower')
    plt.tight_layout()


def plotzgrp(zhist, zedges=None, stretch=4, sort=False, figsize=(10,5)):
    ngrp, nzbin = zhist.shape
    if zedges is None:
        zc = np.arange(nzbin)
    else:
        zc = 0.5 * (zedges[1:] + zedges[:-1])
    zplot = zhist / zhist.max(axis=1, keepdims=True)
    if sort:
        zavg = np.sum(zhist * np.arange(nzbin), axis=1) / np.sum(zhist, axis=1)
        zplot = zplot[np.argsort(zavg)]
    fig = plt.figure(figsize=figsize)
    yoffsets = np.arange(ngrp) / stretch
    for dndz, dy in zip(zplot[::-1], yoffsets[::-1]):
        #plt.plot(zc, dndz + dy, 'r-', alpha=0.5)
        plt.fill_between(zc, dndz + dy, dy, facecolor=(1, 1, 1, 0.75), edgecolor=(1, 0, 0, 0.5), lw=1)
    plt.gca().axis('off')
    plt.tight_layout()
