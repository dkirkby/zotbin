import hashlib

import numpy as np


def prepare(data, bands, useflux='i'):
    """Prepare features from input data and flag galaxies with undetected flux.
    """
    if data.shape[1] != len(bands):
        raise ValueError(f'data has {data.shape[1]} columns but expected {len(bands)} for {bands}.')
    # Flag any galaxies with undetected flux.
    detected = np.all((data != 30) & np.isfinite(data), axis=1)
    print(f'Found {np.count_nonzero(~detected)} galaxies with undetected flux in at least one band.')
    # Use colors plus one flux as the input features.
    colors = np.diff(data, axis=1)
    if useflux not in bands:
        raise ValueError('requested band "{useflux}" not available in "{bands}".')
    idx = bands.index(useflux)
    flux = data[:, idx:idx+1]
    features = np.concatenate((colors, flux), axis=1)
    return features, detected


def get_signature(X):
    # Build a ~unique signature for the input dataset.
    signature = np.concatenate((X.shape, np.argmin(X, axis=0), np.argmax(X, axis=0)))
    return hashlib.sha256(signature).hexdigest()
