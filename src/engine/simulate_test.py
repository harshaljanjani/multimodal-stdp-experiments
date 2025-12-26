import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.loader import load_config
from src.network.builder import build_network
from src.engine.simulate import Simulator
import matplotlib.pyplot as plt

# NOTE: increase input → E synaptic weight mean (e.g., normal, mean = 5.0) to boost excitatory drive
# raise fixed-probability topology from 0.05 to 0.2 to overstimulate inhibitory neurons
def main():
    print("=== RUN EXPERIMENT ===")
    base_path = Path(__file__).parent.parent.parent
    # network_config_path = base_path / "config" / "network" / "e_i_balance.json"
    network_config_path = base_path / "config" / "network" / "spatial_network.json"
    sim_config_path = base_path / "config" / "simulation" / "default_run.json"
    print(f"Loading network config: {network_config_path}")
    network_config = load_config(network_config_path)
    print(f"Loading simulation config: {sim_config_path}")
    sim_config = load_config(sim_config_path)
    print("Building network...")
    network = build_network(network_config)
    print(f"Network built with {network['membrane_potential'].shape[0]} neurons and {network['source_neurons'].shape[0]} synapses.")
    print("Initializing simulator...")
    simulator = Simulator(network, sim_config)
    print("Running simulation...")
    spike_times, spike_indices = simulator.run()
    print("Simulation finished.")
    if len(spike_times) > 0:
        print(f"Total spikes recorded: {len(spike_times)}")
        print("Generating raster plot...")
        plt.style.use('dark_background')
        _, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(spike_times, spike_indices, s=0.5, c='cyan', marker='|', alpha=0.8)
        ax.set_title("Network Activity (Raster Plot)", fontsize=16)
        ax.set_xlabel("Time (ms)", fontsize=12)
        ax.set_ylabel("Neuron ID", fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray')
        plt.show()
    else:
        print("No spikes were recorded.")

main()