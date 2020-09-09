import itertools
import functools

import numpy as np

import scipy.special

from sklearn import preprocessing

import jax
import jax.numpy as jnp
import jax.experimental.stax as stax
import jax.experimental.optimizers as optimizers

import flows


def get_masks(input_dim, hidden_dim, num_hidden):
    masks = []
    input_degrees = jnp.arange(input_dim)
    degrees = [input_degrees]

    for n_h in range(num_hidden + 1):
        degrees += [jnp.arange(hidden_dim) % (input_dim - 1)]
    degrees += [input_degrees % input_dim - 1]
    #print('degrees:', [d.shape for d in degrees])

    for (d0, d1) in zip(degrees[:-1], degrees[1:]):
        masks += [jnp.transpose(jnp.expand_dims(d1, -1) >= jnp.expand_dims(d0, 0)).astype(jnp.float32)]
    #print('masks:', [m.shape for m in masks])
    return masks


def masked_transform(rng, input_dim, hidden_dim, num_hidden):
    masks = get_masks(input_dim, hidden_dim=hidden_dim, num_hidden=num_hidden)
    act = stax.Relu
    stack = [flows.MaskedDense(masks[0]), act]
    for i in range(num_hidden):
        stack += [flows.MaskedDense(masks[i + 1]), act]
    stack.append(flows.MaskedDense(masks[-1].tile(2)))
    init_fun, apply_fun = stax.serial(*stack)
    _, params = init_fun(rng, (input_dim,))
    return params, apply_fun


def learn_flow(X, hidden_dim=48, num_hidden=2, num_unit=5, num_epochs=100, batch_size=2000, interval=None, seed=123):
    """
    """
    # Preprocess input data.
    scaler = preprocessing.StandardScaler().fit(X)
    X_preproc = scaler.transform(X)
    input_dim = X.shape[1]

    # Initialize random numbers.
    rng, flow_rng = jax.random.split(jax.random.PRNGKey(seed))

    # Initialize our flow bijection.
    transform = functools.partial(masked_transform, hidden_dim=hidden_dim, num_hidden=num_hidden)
    bijection_init_fun = flows.Serial(*(flows.MADE(transform), flows.Reverse()) * num_unit)

    # Create direct and inverse bijection functions.
    rng, bijection_rng = jax.random.split(rng)
    bijection_params, bijection_direct, bijection_inverse = bijection_init_fun(bijection_rng, input_dim)

    # Initialize our flow model.
    prior_init_fun = flows.Normal()
    flow_init_fun = flows.Flow(bijection_init_fun, prior_init_fun)
    initial_params, log_pdf, sample = flow_init_fun(flow_rng, input_dim)

    if interval is not None:
        import matplotlib.pyplot as plt
        bins = np.linspace(-1.5, 1.5, 101)

    def loss_fn(params, inputs):
        return -log_pdf(params, inputs).mean()

    @jax.jit
    def step(i, opt_state, inputs):
        params = get_params(opt_state)
        loss_value, gradients = jax.value_and_grad(loss_fn)(params, inputs)
        return opt_update(i, gradients, opt_state), loss_value

    opt_init, opt_update, get_params = optimizers.adam(step_size=1e-3)
    opt_state = opt_init(initial_params)
    root2 = np.sqrt(2.)

    itercount = itertools.count()
    for epoch in range(num_epochs):
        rng, permute_rng = jax.random.split(rng)
        X_epoch = jax.random.permutation(permute_rng, X_preproc)
        for batch_index in range(0, len(X), batch_size):
            opt_state, loss = step(next(itercount), opt_state, X_epoch[batch_index:batch_index+batch_size])
        if interval is not None and (epoch + 1) % interval == 0:
            print(f'epoch {epoch + 1} loss {loss:.3f}')
            # Map the input data back through the flow to the prior space.
            epoch_params = get_params(opt_state)
            X_normal, _ = bijection_direct(epoch_params, X_epoch)
            X_uniform = 0.5 * (1 + scipy.special.erf(np.array(X_normal, np.float64) / root2))
            for i in range(input_dim):
                plt.hist(X_uniform[:,i], bins, histtype='step')
            plt.show()

    # Return a function that maps samples to a ~uniform distribution on [0,1] ** input_dim.
    # Takes a numpy array as input and returns a numpy array of the same shape.
    final_params = get_params(opt_state)
    def flow_map(Y):
        Y_preproc = scaler.transform(Y)
        Y_normal, _ = bijection_direct(final_params, jnp.array(Y_preproc))
        return np.array(Y_normal)
        #Y_uniform = 0.5 * (1 + scipy.special.erf(np.array(Y_normal, np.float64) / root2))
        #return Y_uniform
    return flow_map
