import cupy as cp
from numba import cuda

@cuda.jit
def _propagate_spikes_kernel(
    spiked_this_step,
    source_neurons,
    target_neurons,
    weights,
    delays,
    spike_buffer,
    current_step
):
    i = cuda.grid(1)
    if i < source_neurons.shape[0]:
        source_id = source_neurons[i]
        if spiked_this_step[source_id] == 1:
            target_id = target_neurons[i]
            delay = delays[i]
            weight = weights[i]
            buffer_size = spike_buffer.shape[0]
            future_step = current_step + delay
            buffer_idx = future_step % buffer_size
            cuda.atomic.add(spike_buffer, (buffer_idx, target_id), weight)

def propagate_spikes(
    spiked_this_step,
    source_neurons,
    target_neurons,
    weights,
    delays,
    spike_buffer,
    current_step
):
    if cp.sum(spiked_this_step) == 0:
        return
    threads_per_block = 256
    blocks_per_grid = (source_neurons.shape[0] + (threads_per_block - 1)) // threads_per_block
    _propagate_spikes_kernel[blocks_per_grid, threads_per_block](
        spiked_this_step,
        source_neurons,
        target_neurons,
        weights,
        delays,
        spike_buffer,
        current_step
    )
