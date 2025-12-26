import cupy as cp
import numpy as np
from typing import Dict, List, Tuple
from src.gpu import n_kernels, s_kernels

class Simulator:
    def __init__(self, network: Dict[str, cp.ndarray], sim_config: Dict):
        self.network = network
        self.dt = sim_config["dt"]
        self.duration = sim_config["duration"]
        self.num_steps = int(self.duration / self.dt)
        self.current_step = 0
        max_delay = int(cp.max(self.network["delays"]).item()) if self.network["delays"].size > 0 else 0
        self.buffer_size = max_delay + 1
        num_neurons = self.network["membrane_potential"].shape[0]
        self.spike_buffer = cp.zeros((self.buffer_size, num_neurons), dtype=cp.float32)

    def run(self) -> Tuple[List[float], List[int]]:
        all_spike_times: List[float] = []
        all_spike_indices: List[np.ndarray] = []
        for step in range(self.num_steps):
            self.current_step = step
            buffer_idx = self.current_step % self.buffer_size
            self.network["membrane_potential"] += self.spike_buffer[buffer_idx, :]
            self.spike_buffer[buffer_idx, :] = 0
            spiked_this_step = n_kernels.update_neurons(
                self.network["membrane_potential"],
                self.network["refractory_time"],
                self.network["v_leak"],
                self.network["v_reset"],
                self.network["v_threshold"],
                self.network["tau_m"],
                self.network["i_background"],
                self.dt
            )
            num_spiked = cp.sum(spiked_this_step).item()
            if num_spiked > 0:
                spiked_indices = cp.where(spiked_this_step == 1)[0]
                all_spike_times.extend([step * self.dt] * num_spiked)
                all_spike_indices.append(cp.asnumpy(spiked_indices))
            s_kernels.propagate_spikes(
                spiked_this_step,
                self.network["source_neurons"],
                self.network["target_neurons"],
                self.network["weights"],
                self.network["delays"],
                self.spike_buffer,
                self.current_step
            )
            if step % 1000 == 0:
                current_time_ms = step * self.dt
                print(f"Time: {current_time_ms:.1f}ms - Spiked this step: {num_spiked}")
        return all_spike_times, np.concatenate(all_spike_indices) if all_spike_indices else np.array([])
