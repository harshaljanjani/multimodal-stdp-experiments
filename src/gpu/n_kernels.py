import cupy as cp
from numba import cuda

@cuda.jit
def _update_neurons_kernel(
    spiked_this_step,
    membrane_potential,
    refractory_time,
    v_leak,
    v_reset,
    v_threshold,
    tau_m,
    i_background,
    dt
):
    i = cuda.grid(1)
    if i < membrane_potential.shape[0]:
        if refractory_time[i] > 0:
            refractory_time[i] -= dt
            membrane_potential[i] = v_reset[i]
        else:
            dv = (v_leak[i] - membrane_potential[i] + i_background[i]) / tau_m[i]
            membrane_potential[i] += dv * dt
            if membrane_potential[i] >= v_threshold[i]:
                spiked_this_step[i] = 1
                membrane_potential[i] = v_reset[i]
                refractory_time[i] = 3.0

def update_neurons(
    membrane_potential,
    refractory_time,
    v_leak,
    v_reset,
    v_threshold,
    tau_m,
    i_background,
    dt
):
    spiked_this_step = cp.zeros_like(membrane_potential, dtype=cp.int8)
    threads_per_block = 256
    blocks_per_grid = (membrane_potential.shape[0] + (threads_per_block - 1)) // threads_per_block
    _update_neurons_kernel[blocks_per_grid, threads_per_block](
        spiked_this_step,
        membrane_potential,
        refractory_time,
        v_leak,
        v_reset,
        v_threshold,
        tau_m,
        i_background,
        dt
    )
    return spiked_this_step
