#!/usr/bin/env python
'''Numba implementation of resampler'''
import numpy as np
import matplotlib.pyplot as plt
import numba


#@numba.jit(nopython=True, nogil=True)
def resample_f(x, y, sample_ratio, interp_win, interp_delta, num_table):
    scale = min(1.0, sample_ratio)
    time_increment = 1. / sample_ratio
    index_step = int(scale * num_table)
    time_register = 0.0

    n = 0
    frac = 0.0
    index_frac = 0.0
    offset = 0
    eta = 0.0
    weight = 0.0

    nwin = interp_win.shape[0]
    n_orig = x.shape[0]
    n_out = y.shape[0]
    n_channels = y.shape[1]

    # Increment the time register
    time_register = np.arange(n_out) * time_increment
    # Grab the top bits as an index to the input buffer
    n = time_register.astype(np.int)
    # Grab the fractional component of the time index
    frac = scale * (time_register - n)
    # Offset into the filter
    index_frac = frac * num_table
    offset = index_frac.astype(np.int)

    i_max = int(np.max((nwin - offset) // index_step))
    interp_win_padded = np.pad(interp_win, (0, 1), 'constant', constant_values=0)
    interp_delta_padded = np.pad(interp_delta, (0, 1), 'constant', constant_values=0)

    # Compute the left wing of the filter response
    # first the interp_win only
    filter_take_idx = offset + np.arange(i_max)[:, None] * index_step
    weight = np.take(interp_win_padded, filter_take_idx, mode='clip')
    x_padded = np.pad(x, ((0, i_max - 1), (0, 0)), 'constant', constant_values=0)
    x_take_idx = n - np.arange(i_max)[:, None]

    #x_take_idx %= x_padded.shape[0]
    x_padded_unfold = np.take(x_padded, x_take_idx, axis=0)


    np.einsum('ij,ijk->jk', weight, x_padded_unfold, out=y, casting='unsafe')

    # delta win
    eta = index_frac - offset
    weight = np.take(interp_delta_padded, filter_take_idx, mode='clip')
    y += np.einsum('ij,j,ijk->jk', weight, eta, x_padded_unfold, casting='unsafe')

    # Invert P
    frac = scale - frac

    # Offset into the filter
    index_frac = frac * num_table
    offset = (index_frac).astype(np.int)



    # Compute the right wing of the filter response
    # first the interp_win only
    k_max = int(np.max((nwin - offset) // index_step))
    filter_take_idx = offset + np.arange(k_max)[:, None] * index_step
    weight = np.take(interp_win_padded, filter_take_idx, mode='clip')

    print(eta)
    x_take_idx = n + 1 + np.arange(k_max)[:, None]
    x_padded_unfold = np.take(x_padded, x_take_idx, axis=0)

    y += np.einsum('ij,ijk->jk', weight, x_padded_unfold, casting='unsafe')

    # delta win
    # Interpolation factor
    eta = index_frac - offset
    weight = np.take(interp_delta_padded, filter_take_idx, mode='clip')
    y += np.einsum('ij,j,ijk->jk', weight, eta, x_padded_unfold, casting='unsafe')


if __name__ == '__main__':

    import resamcupy
    sr = 22050
    total_time = 2
    x = np.arange(sr * total_time)
    y = np.sin(np.cumsum(2 * np.pi  * np.linspace(0.0001, 0.01, sr * total_time)))

    y2 = resamcupy.resample(y, sr, sr * 1.4)


    plt.plot(y)
    plt.plot(y2)
    plt.show()