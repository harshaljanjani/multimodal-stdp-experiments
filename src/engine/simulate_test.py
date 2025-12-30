import sys
from pathlib import Path
import numpy as np
import cupy as cp

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.loader import load_config
from src.network.builder import build_network
from src.engine.simulate import Simulator
import matplotlib.pyplot as plt

# NOTE: increase input → E synaptic weight mean (e.g., normal, mean = 5.0) to boost excitatory drive
# raise fixed-probability topology from 0.05 to 0.2 to overstimulate inhibitory neurons
def run_experiment(network_config_name: str, base_path: Path):
    network_config_path = base_path / "config" / "network" / network_config_name
    sim_config_path = base_path / "config" / "simulation" / "default_run.json"
    print(f"\n=== RUNNING EXPERIMENT: {network_config_name} ===")
    print(f"Loading network config: {network_config_path}")
    network_config = load_config(network_config_path)
    print(f"Loading simulation config: {sim_config_path}")
    sim_config = load_config(sim_config_path)
    print("Building network...")
    network, pop_info = build_network(network_config)
    print(f"Network built with {network['membrane_potential'].shape[0]} neurons and {network['source_neurons'].shape[0]} synapses.")
    print("Initializing simulator...")
    simulator = Simulator(network, pop_info, sim_config)
    print("Running simulation...")
    spike_times, spike_indices = simulator.run()
    print("Simulation finished.")
    if len(spike_times) > 0:
        print(f"Total spikes recorded: {len(spike_times)}")
        plt.style.use('dark_background')
        _, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(spike_times, spike_indices, s=0.5, c='cyan', marker='|', alpha=0.8)
        ax.set_title(f"Network Activity: {network_config_name}", fontsize=16)
        ax.set_xlabel("Time (ms)", fontsize=12)
        ax.set_ylabel("Neuron ID", fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray')
        plt.show()
    else:
        print("No spikes were recorded for this configuration.")

def run_embodiment_test(base_path: Path):
    print("\n=== RUNNING 'REFLEX ARC' TEST ===")
    reflex_arc_config_path = base_path / "config" / "network" / "reflex_arc.json"
    sim_config_path = base_path / "config" / "simulation" / "default_run.json"
    sim_config = load_config(sim_config_path)
    sim_config['duration'] = 500    #  run 500 ms
    print("\nBuilding reflex arc network...")
    reflex_arc_config = load_config(reflex_arc_config_path)
    network, pop_info = build_network(reflex_arc_config)
    print(f"Network built with {network['membrane_potential'].shape[0]} neurons and {network['source_neurons'].shape[0]} synapses.")
    simulator = Simulator(network, pop_info, sim_config)
    print("\nRunning brainstem simulation loop...")
    light_on = False
    for step in range(simulator.num_steps):
        # turn light on at 200ms; testing in a sandbox env for now.
        if step * simulator.dt >= 200.0:
            light_on = True
        sensory_spikes = None
        if light_on:
            eye_pop = pop_info['Left_Eye']
            num_firing = int(eye_pop['count'] * 0.2)
            firing_indices = np.random.choice(np.arange(eye_pop['start'], eye_pop['end']), size=num_firing, replace=False)
            sensory_spikes = cp.asarray(firing_indices, dtype=cp.int32)
        simulator.step(sensory_spikes_indices=sensory_spikes)
        if step > 0 and step % 200 == 0:
            motor_rates = simulator.get_motor_firing_rates()
            motor_rate = motor_rates.get('Turn_Left_Motor', 0.0)
            status = "ON" if light_on else "OFF"
            print(f"Time: {step * simulator.dt:.1f}ms | Light Status: {status} | Motor Firing Rate: {motor_rate:.2f} Hz")
    print("\nEmbodiment test finished.")

def main():
    base_path = Path(__file__).resolve().parent.parent.parent
    run_embodiment_test(base_path)
    configurations_to_test = [
        "pruning_test.json",
        "learning_test.json",
        "e_i_balance.json",
        "spatial_network.json"
    ]
    for config_name in configurations_to_test:
        try:
            run_experiment(config_name, base_path)
        except Exception as e:
            print(f"\n=== ERROR running {config_name} ===")
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()