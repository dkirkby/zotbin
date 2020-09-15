import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental import stax, optimizers

from zotbin.reweight import reweighted_metrics


def create_model(nbin, nhidden, nlayer):
    layers = []
    for i in range(nlayer):
        layers.extend([
            stax.Dense(nhidden),
            stax.LeakyRelu,
            stax.BatchNorm(axis=(0,1)),
        ])
    layers.extend([
        stax.Dense(nbin),
        stax.Softmax
    ])
    return stax.serial(*layers)


def learn_nnet(nbin, X, z, init_data, trainfrac=0.5, nhidden=64, nlayer=2, ntrial=1, nepoch=100,
               batchsize=50000, interval=10, eta=0.001, metric='FOM_DETF_3x2', gals_per_arcmin2=20., fsky=0.25, seed=123):
    """
    """
    zedges = np.array(init_data[0])
    # Include any overflow in the last bin.
    zedges[-1] = max(zedges[-1], z.max() + 1e-3)
    zedges = jnp.array(zedges)
    # Calculate the normalized dndz for each bin.
    def hist1bin(z, w):
        return jnp.histogram(z, bins=zedges, weights=w)[0]
    histbins = jax.vmap(hist1bin, in_axes=(None, 1), out_axes=0)

    # Build the network model.
    init_fun, apply_fun = create_model(nbin, nhidden, nlayer)

    def get_dndz(params, Xb, zb):
        # Get normalized weights from the network model.
        wgts = apply_fun(params, Xb)
        # Convert to a normalized dndz for each bin.
        w = histbins(zb, wgts)
        return w / zb.shape[0]

    # Evaluate all 3x2 metrics.
    def metrics(params, Xb, zb):
        w = get_dndz(params, Xb, zb)
        # Use the same binning for number density and weak lensing.
        weights = jnp.array([w, w])
        return reweighted_metrics(weights, *init_data[1:], gals_per_arcmin2, fsky)

    # Evaluate the loss as -metric.
    def loss(params, Xb, zb):
        return -metrics(params, Xb, zb)[metric]

    opt_init, opt_update, get_params = optimizers.adam(eta)

    # Define the update rule.
    @jax.jit
    def update(i, opt_state, Xb, zb):
        params = get_params(opt_state)
        grads = jax.grad(loss)(params, Xb, zb)
        return opt_update(i, grads, opt_state)

    rng = jax.random.PRNGKey(seed)
    gen = np.random.RandomState(seed)

    ndata = X.shape[0]
    ntrain = int(np.round(ndata * trainfrac))
    Xtrain, ztrain = X[:ntrain], z[:ntrain]
    Xvalid, zvalid = X[ntrain:], z[ntrain:]
    nbatch = ntrain // batchsize

    all_scores = []
    max_score = -1
    best_params = None

    for trial in range(ntrial):

        _, init_params = init_fun(rng, (batchsize, X.shape[1]))
        opt_state = opt_init(init_params)

        all_scores.append([])
        itercount = itertools.count()
        for epoch in range(nepoch):
            perm = gen.permutation(ntrain)
            for batch in range(nbatch):
                idx = perm[batch * batchsize:(batch + 1) * batchsize]
                opt_state = update(next(itercount), opt_state, Xtrain[idx], ztrain[idx])
            # Calculate train and validation scores after this epoch.
            params = get_params(opt_state)
            train_score = metrics(params, Xtrain, ztrain)[metric]
            all_scores[-1].append(train_score)
            if train_score > max_score:
                best_params = params
                max_score = train_score
            if epoch % interval == 0:
                valid_score = metrics(params, Xvalid, zvalid)[metric]
                print(f'Trial {trial+1}/{ntrial} epoch {epoch+1}/{nepoch} train {train_score:.3f} (max {max_score:.3f}) validation {valid_score:.3f}.')

    # Calculate all metrics for the best params.
    best_scores = metrics(best_params, Xtrain, ztrain)
    best_scores = {metric: float(value) for metric, value in best_scores.items()}

    # Calculate the best weights.
    weights = apply_fun(best_params, Xtrain)

    # Convert the best parameters to normalized dndz weights.
    dndz_bin = get_dndz(best_params, Xtrain, ztrain)

    apply = lambda X: apply_fun(best_params, X)

    return best_scores, weights, dndz_bin, all_scores, apply
