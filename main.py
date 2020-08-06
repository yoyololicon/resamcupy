import numpy as np
from numba import cuda, jit, float32, float64
from math import sin, pi, cos, copysign

TPB = 256
SHARED_SIZE = 2048

def resample_f(x, y, sample_ratio, num_zeros, roll_off):
    scale = min(1.0, sample_ratio)
    time_increment = 1. / sample_ratio

    block = y.shape[0] // TPB + 1
    memsize = int(TPB * time_increment) + int(num_zeros / scale) * 2 + 1
    assert memsize < SHARED_SIZE
    resample_cuda[TPB, block](x, y, num_zeros, scale, time_increment, roll_off)

@cuda.jit(device=True)
def sinc(x):
    x = copysign(1, x) * max(abs(x), 1e-20) * pi
    return sin(x) / x

@cuda.jit(device=True)
def hann(n, M):
    return 0.5 - 0.5 * cos(pi * n / M)


#@cuda.jit(device=True)
def blackmanharris(n, M):
    x = 2 * pi * n / (M - 1)
    return 0.35875 - 0.48892 * cos(x) + 0.14128 * cos(2 * x) - 0.01168 * cos(3 * x)
#
#@jit(nopython=True)
@cuda.jit(debug=False)
def resample_cuda(x, y, num_zeros, scale, time_increment, roll_off):
    sX = cuda.shared.array(shape=SHARED_SIZE, dtype=float32)
    lw = cuda.local.array(shape=128, dtype=float32)
    TPB = cuda.blockDim.x
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x

    n_orig = x.shape[0]
    n_out = y.shape[0]
    pos = cuda.grid(1)

    mem_offset = int(num_zeros / scale) + 1
    memsize = int(TPB * time_increment) + mem_offset * 2
    t_step = int(memsize / TPB) + 1
    t_start_idx = t_step * tx
    blockX_n = int(bx * TPB * time_increment)
    if t_start_idx < memsize:
        for i in range(t_start_idx, min(memsize, t_start_idx + t_step)):
            sX[i] = x[min(n_orig - 1, max(0, blockX_n - mem_offset + i))]

    cuda.syncthreads()

    if pos < n_out:
        time_register = pos * time_increment
        n = int(time_register)

        # Grab the fractional component of the time index
        frac = scale * (time_register - n)
        index_step = scale
        M = num_zeros * roll_off
        tmp = 0.0

        i_max = min(n + 1, int((num_zeros - frac) / index_step))
        for i in range(i_max):
            sinc_pos = (frac + i * index_step) * roll_off
            weight = sinc(sinc_pos) * blackmanharris(sinc_pos + M, M)
            lw[i] = weight
            #tmp += weight * x[n - i]
            tmp += weight * sX[n - blockX_n + mem_offset - i]

        frac = scale - frac

        k_max = min(n_orig - n - 1, int((num_zeros - frac) / index_step))
        for k in range(k_max):
            sinc_pos = (frac + k * index_step) * roll_off
            weight = sinc(sinc_pos) * blackmanharris(sinc_pos + M, M)
            lw[i] = weight
            #tmp += weight * x[n + k + 1]
            tmp += weight * sX[n - blockX_n + mem_offset + k + 1]

        tmp *= min(1 / time_increment, 1) * roll_off
        y[pos] = tmp


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import resamcupy

    print(cuda.is_available())

    def make_tone(freq, sr, duration):
        return np.sin(2 * np.pi * freq / sr * np.arange(int(sr * duration)))

    def make_sweep(freq, sr, duration):
        return np.sin(np.cumsum(2 * np.pi * np.logspace(np.log2(2.0 / sr),
                                                        np.log2(float(freq) / sr),
                                                        num=int(duration * sr), base=2.0)))

    window = []

    for i in range(45):
        window.append(blackmanharris(i, 45))
    print(window[0], window[-1])

    x = make_sweep(100, 22050, 3)
    x = make_tone(512, 22050, 2)
    ref = make_tone(512, 44100, 2)
    xt = np.arange(x.shape[0]) / 22050
    #y = resampy.resample(x, 22050, 44100, filter='sinc_window', num_zeros=16, rolloff=0.85)
    #y = np.empty(int(x.shape[0] * 1000 / 4410))
    y = resamcupy.resample_sinc(x, 22050, 44100, num_zeros=32, rolloff=0.85)
    #y = cuda.to_device(yhost)
    #resample_f(cuda.to_device(x), y, 1000 / 4410, 32, 0.95)
    #y.copy_to_host(yhost)
    yt = np.arange(y.shape[0]) / 44100
    #plt.plot(xt, x)
    plt.plot(yt, y - ref)
    plt.show()

    # x = np.arange(50).reshape(50, 1)
    # x = cuda.to_device(x)
    # y = cuda.device_array_like(np.zeros(100).reshape(100, 1))

    # resample_cuda[(5, 1), (24, 1)](x, y, 64, 2, 0.5, 0.9)
    # print(y.copy_to_host(), x.copy_to_host())
