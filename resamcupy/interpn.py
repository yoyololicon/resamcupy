#!/usr/bin/env python
'''Numba implementation of resampler'''
import cupy as cp
import numba


#@numba.jit
def resample_f(x, y, sample_ratio, interp_win, interp_delta, num_table):
    scale = min(1.0, sample_ratio)
    time_increment = 1. / sample_ratio
    index_step = int(scale * num_table)

    nwin = interp_win.shape[0]
    n_orig = x.shape[0]
    n_out = y.shape[0]
    #n_channels = y.shape[1]

    xp = cp.get_array_module(x)

    # Increment the time register
    time_register = xp.arange(n_out) * time_increment
    # Grab the top bits as an index to the ixput buffer
    n = time_register.astype(xp.int)
    # Grab the fractional component of the time index
    frac = scale * (time_register - n)
    # Offset into the filter
    index_frac = frac * num_table
    offset = index_frac.astype(xp.int)

    i_max = nwin // index_step
    interp_win_padded = xp.pad(interp_win, (0, 1), 'constant', constant_values=0)
    x_padded = xp.pad(x, ((0, i_max - 1),) +  ((0, 0),) * (len(x.shape) - 1), 'constant', constant_values=0)

    # Compute the left wing of the filter response
    # first the interp_win only
    filter_take_idx = offset + xp.arange(i_max)[:, None] * index_step
    xp.minimum(filter_take_idx, interp_win_padded.shape[0] - 1, out=filter_take_idx)
    weight = xp.take(interp_win_padded, filter_take_idx)
    x_take_idx = n - xp.arange(i_max)[:, None]

    # x_take_idx %= x_padded.shape[0]
    x_padded_unfold = xp.take(x_padded, x_take_idx, axis=0)

    xp.add(y, xp.einsum('ij,ij...->j...', weight, x_padded_unfold, optimize=True), out=y, casting='unsafe')

    # delta win
    eta = index_frac - offset
    xp.minimum(filter_take_idx, interp_delta.shape[0] - 1, out=filter_take_idx)
    weight = xp.take(interp_delta, filter_take_idx)
    xp.add(y, xp.einsum('ij,j,ij...->j...', weight, eta, x_padded_unfold, optimize=True), out=y, casting='unsafe')

    # Invert P
    frac = scale - frac

    # Offset into the filter
    index_frac = frac * num_table
    offset = (index_frac).astype(xp.int)

    # Compute the right wing of the filter response
    # first the interp_win only
    k_max = i_max - 1
    filter_take_idx = offset + xp.arange(k_max)[:, None] * index_step
    xp.minimum(filter_take_idx, interp_win_padded.shape[0] - 1, out=filter_take_idx)
    weight = xp.take(interp_win_padded, filter_take_idx)

    x_take_idx = n + 1 + xp.arange(k_max)[:, None]
    x_padded_unfold = xp.take(x_padded, x_take_idx, axis=0)
    xp.add(y, xp.einsum('ij,ij...->j...', weight, x_padded_unfold, optimize=True), out=y, casting='unsafe')

    # delta win
    # Interpolation factor
    eta = index_frac - offset
    xp.minimum(filter_take_idx, interp_delta.shape[0] - 1, out=filter_take_idx)
    weight = xp.take(interp_delta, filter_take_idx)
    xp.add(y, xp.einsum('ij,j,ij...->j...', weight, eta, x_padded_unfold, optimize=True), out=y, casting='unsafe')
