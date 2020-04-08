#!/usr/bin/env python
'''Numba implementation of resampler'''
import cupy as cp
from numba import cuda


def resample_f(x, y, sample_ratio, interp_win, interp_delta, num_table):
    scale = min(1.0, sample_ratio)
    time_increment = 1. / sample_ratio
    index_step = int(scale * num_table)

    TPB = 256
    block = y.shape[0] // TPB + 1
    resample_cuda[(block, 1), (TPB, 1)](x, y, interp_win, interp_delta, num_table, scale, time_increment, index_step)


@cuda.jit
def resample_cuda(x, y, interp_win, interp_delta, num_table, scale, time_increment, index_step):
    nwin = interp_win.shape[0]
    n_orig = x.shape[0]
    n_out = y.shape[0]
    n_channels = y.shape[1]

    pos, _ = cuda.grid(2)
    if pos < n_out:
        time_register = pos * time_increment
        n = int(time_register)

        # Grab the fractional component of the time index
        frac = scale * (time_register - n)

        # Offset into the filter
        index_frac = frac * num_table
        offset = int(index_frac)

        # Interpolation factor
        eta = index_frac - offset

        # Compute the left wing of the filter response
        i_max = min(n + 1, (nwin - offset) // index_step)
        for i in range(i_max):
            weight = interp_win[offset + i * index_step] + eta * interp_delta[offset + i * index_step]
            for j in range(n_channels):
                y[pos, j] += weight * x[n - i, j]

        # Invert P
        frac = scale - frac

        # Offset into the filter
        index_frac = frac * num_table
        offset = int(index_frac)

        # Interpolation factor
        eta = index_frac - offset

        # Compute the right wing of the filter response
        k_max = min(n_orig - n - 1, (nwin - offset) // index_step)
        for k in range(k_max):
            weight = interp_win[offset + k * index_step] + eta * interp_delta[offset + k * index_step]
            for j in range(n_channels):
                y[pos, j] += weight * x[n + k + 1, j]

