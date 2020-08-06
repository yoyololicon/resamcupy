from numba import cuda, jit, float32, float64
from math import sin, pi, cos, copysign

MIN_TPB = 256
MAX_TPB = 1024
MAX_SHARED_SIZE = 8000


def resample_f(x, y, sample_ratio, num_zeros, roll_off):
    scale = min(1.0, sample_ratio)
    time_increment = 1. / sample_ratio

    tmp_memsize = int(MAX_TPB * time_increment) + int(num_zeros / scale) * 2 + 1
    if tmp_memsize > MAX_SHARED_SIZE:
        TPB = int((MAX_SHARED_SIZE - 1 - int(num_zeros / scale) * 2) / time_increment)
        memsize = MAX_SHARED_SIZE
    else:
        TPB = MAX_TPB
        memsize = tmp_memsize

    block, _ = divmod(y.shape[0], TPB)
    if _:
        block += 1

    resample_cuda[(block, y.shape[1]), (TPB, 1), 0, memsize * 4](x, y, num_zeros, scale, time_increment, roll_off)


@cuda.jit(device=True)
def sinc(x):
    x = copysign(1, x) * max(abs(x), 1e-20) * pi
    return sin(x) / x


@cuda.jit(device=True)
def hann(n, N):
    return 0.5 - 0.5 * cos(2 * pi * n / (N - 1))


@cuda.jit(device=True)
def hamming(n, N):
    return 0.54 - 0.46 * cos(2 * pi * n / (N - 1))


@cuda.jit(device=True)
def blackman(n, N):
    x = 2 * pi * n / (N - 1)
    return 0.42 - 0.5 * cos(x) + 0.08 * cos(2 * x)  # - 0.01168 * cos(3 * x)


@cuda.jit(device=True)
def blackmanharris(n, N):
    x = 2 * pi * n / (N - 1)
    return 0.355768 - 0.487396 * cos(x) + 0.144232 * cos(2 * x) - 0.01168 * cos(3 * x)



@cuda.jit
def resample_cuda(x, y, num_zeros, scale, time_increment, roll_off):
    sX = cuda.shared.array(shape=0, dtype=float32)
    TPB = cuda.blockDim.x
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x

    n_orig = x.shape[0]
    n_out = y.shape[0]
    pos, c = cuda.grid(2)
    n = 0
    i_max = k_max = 0

    mem_offset = int(num_zeros / scale) + 1
    memsize = int(TPB * time_increment) + mem_offset * 2
    t_step = int(memsize / TPB) + 1
    t_start_idx = t_step * tx
    blockX_n = int(bx * TPB * time_increment)

    if t_start_idx < memsize:
        for i in range(t_start_idx, min(memsize, t_start_idx + t_step)):
            sX[i] = x[min(n_orig - 1, max(0, blockX_n - mem_offset + i)), c]

    cuda.syncthreads()

    if pos < n_out:
        time_register = pos * time_increment
        n = int(time_register)

        # Grab the fractional component of the time index
        frac = scale * (time_register - n)
        index_step = scale
        M = num_zeros * roll_off
        tmp = 0.

        i_max = min(n + 1, int((num_zeros - frac) / index_step))
        for i in range(i_max):
            sinc_pos = (frac + i * index_step)
            weight = sinc(sinc_pos * roll_off) * hann(num_zeros - sinc_pos, num_zeros * 2 + 1)
            tmp += weight * sX[n - blockX_n + mem_offset - i]

        frac = scale - frac
        k_max = min(n_orig - n - 1, int((num_zeros - frac) / index_step))
        for k in range(k_max):
            sinc_pos = (frac + k * index_step)
            weight = sinc(sinc_pos * roll_off) * hann(sinc_pos + num_zeros, num_zeros * 2 + 1)
            tmp += weight * sX[n - blockX_n + mem_offset + k + 1]

        y[pos, c] = tmp * scale * roll_off
