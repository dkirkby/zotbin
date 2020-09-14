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
    """Build a ~unique signature for the numpy array X suitable for cache file names.
    """
    signature = np.concatenate((X.shape, np.argmin(X, axis=0), np.argmax(X, axis=0)))
    return hashlib.sha256(signature).hexdigest()


def get_file(name, destdir='.',
             url='https://portal.nersc.gov/cfs/lsst/dkirkby/zotbin/',
             nersc_path='/global/cfs/cdirs/lsst/www/dkirkby/zotbin',
             progress_report_frac=0.1):
    """Locate the named file in the destination directory.

    If necessary, download via HTTP or, if running at NERSC, make a softlink.
    """
    dest = os.path.join(destdir, name)
    if not os.path.exists(dest):
        if os.environ.get('NERSC_HOST'):
            src = os.path.join(nersc_path, name)
            if not os.path.exists(src):
                raise ValueError(f'File not found at NERSC: {src}')
            os.symlink(src, dest)
            print(f'Created a softlink to {src}')
        else:
            src = os.path.join(url, name)
            print(f'Downloading {src}...')
            next_report_frac = progress_report_frac
            def progress(block_num, block_size, total_size):
                nonlocal next_report_frac
                frac = block_num * block_size / total_size
                if frac >= next_report_frac:
                    print(f'...downloaded {100 * frac:.0f}%', flush=True)
                    next_report_frac += progress_report_frac
            urlretrieve(src, dest, reporthook=progress)
    assert os.path.exists(dest)
    return dest
