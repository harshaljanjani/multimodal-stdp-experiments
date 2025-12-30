import cupy as cp
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from src.gpu import n_kernels, s_kernels, stdp_kernels

class Simulator:
    def __init__(self, network: Dict[str, cp.ndarray], pop_info: Dict[str, Any], sim_config: Dict):
        self.network = network
        self.pop_info = pop_info
        self.dt = sim_config["dt"]
        self.duration = sim_config["duration"]
        self.num_steps = int(self.duration / self.dt)
        self.current_step = 0
        self.pruning_interval = sim_config.get("pruning_interval", 0)
        self._initialize_spike_buffer()
        self._initialize_motor_readout()

    def _initialize_spike_buffer(self):
        max_delay = int(cp.max(self.network["delays"]).item()) if self.network["delays"].size > 0 else 0
        self.buffer_size = max_delay + 1
        num_neurons = self.network["membrane_potential"].shape[0]
        self.spike_buffer = cp.zeros((self.buffer_size, num_neurons), dtype=cp.float32)

    def _initialize_motor_readout(self):
        self.motor_populations = {
            name: info for name, info in self.pop_info.items() if info.get('type') == 'motor'
        }
        self.motor_readout_window_steps = 100  # 10 ms window if dt = 0.1 ms
        self.motor_spike_history = {
            name: cp.zeros(self.motor_readout_window_steps, dtype=cp.int32)
            for name in self.motor_populations
        }

    def _prune_synapses(self):
        num_synapses_before = self.network['source_neurons'].shape[0]
        if num_synapses_before == 0:
            return
        keep_mask = cp.abs(self.network['weights']) >= self.network['prune_threshold']
        num_synapses_after = int(cp.sum(keep_mask).item())
        if num_synapses_after < num_synapses_before:
            print(f"\n=== Pruning ===")
            print(f"Time: {self.current_step * self.dt:.1f}ms - Synapses before: {num_synapses_before}")
            synaptic_keys = [
                "source_neurons", "target_neurons", "weights", "delays",
                "learning_rate", "max_weight", "prune_threshold"
            ]
            for key in synaptic_keys:
                if key in self.network:
                    self.network[key] = self.network[key][keep_mask]
            print(f"Pruned {num_synapses_before - num_synapses_after} synapses. Synapses after: {num_synapses_after}")
            self._initialize_spike_buffer()
            print(f"=== Pruning Complete ===\n")

    def step(self, sensory_spikes_indices: Optional[cp.ndarray] = None) -> cp.ndarray:
        # periodically pruned.
        if self.pruning_interval > 0 and self.current_step > 0 and self.current_step % self.pruning_interval == 0:
            self._prune_synapses()
        buffer_idx = self.current_step % self.buffer_size
        self.network["membrane_potential"] += self.spike_buffer[buffer_idx, :]
        self.spike_buffer[buffer_idx, :] = 0    
        spiked_this_step = n_kernels.update_neurons(
            self.network["membrane_potential"], self.network["refractory_time"],
            self.network["v_leak"], self.network["v_reset"], self.network["v_threshold"],
            self.network["tau_m"], self.network["i_background"], self.dt
        )
        if sensory_spikes_indices is not None:
            spiked_this_step[sensory_spikes_indices] = 1
        num_spiked = cp.sum(spiked_this_step).item()
        history_idx = self.current_step % self.motor_readout_window_steps
        for name, info in self.motor_populations.items():
            start, end = info['start'], info['end']
            spike_count = cp.sum(spiked_this_step[start:end]).item()
            self.motor_spike_history[name][history_idx] = spike_count
        if num_spiked > 0 and self.network['source_neurons'].shape[0] > 0:
            stdp_kernels.update_weights(
                self.network["weights"], self.network["source_neurons"], self.network["target_neurons"],
                spiked_this_step, self.network["trace_pre"], self.network["trace_post"],
                self.network["learning_rate"], self.network["max_weight"]
            ) 
        stdp_kernels.update_traces(
            self.network["trace_pre"], self.network["trace_post"], spiked_this_step,
            self.network["tau_trace_pre"], self.network["tau_trace_post"], self.dt
        )
        if num_spiked > 0 and self.network['source_neurons'].shape[0] > 0:
            s_kernels.propagate_spikes(
                spiked_this_step, self.network["source_neurons"], self.network["target_neurons"],
                self.network["weights"], self.network["delays"], self.spike_buffer, self.current_step
            )
        self.current_step += 1
        return spiked_this_step

    def get_motor_firing_rates(self) -> Dict[str, float]:
        rates = {}
        total_time_s = (self.motor_readout_window_steps * self.dt) / 1000.0
        if total_time_s == 0:
            return {name: 0.0 for name in self.motor_populations}
        for name, history in self.motor_spike_history.items():
            total_spikes = cp.sum(history).item()
            num_neurons = self.motor_populations[name]['count']
            avg_rate = (total_spikes / num_neurons) / total_time_s if num_neurons > 0 else 0.0
            rates[name] = avg_rate
        return rates

    def run(self) -> Tuple[List[float], List[int]]:
        all_spike_times: List[float] = []
        all_spike_indices: List[np.ndarray] = []
        for step_num in range(self.num_steps):
            spiked_this_step = self.step()
            num_spiked = cp.sum(spiked_this_step).item()
            if num_spiked > 0:
                spiked_indices = cp.where(spiked_this_step == 1)[0]
                all_spike_times.extend([step_num * self.dt] * num_spiked)
                all_spike_indices.append(cp.asnumpy(spiked_indices))
            if step_num % 1000 == 0:
                current_time_ms = step_num * self.dt
                print(f"Time: {current_time_ms:.1f}ms - Spiked this step: {num_spiked}")
        # print(f"Final weight: {self.network['weights'][0]}")
        print("\nSimulation finished.")
        return all_spike_times, np.concatenate(all_spike_indices) if all_spike_indices else np.array([])
