import sys
from pathlib import Path
import cupy as cp
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.loader import load_config
from src.network.builder import build_network
from src.engine.simulate import Simulator

def get_synapse_indices(network, pop_info, source_pop_name, target_pop_name):
    source_pop = pop_info[source_pop_name]
    target_pop = pop_info[target_pop_name]
    source_mask = (network["source_neurons"] >= source_pop["start"]) & (network["source_neurons"] < source_pop["end"])
    target_mask = (network["target_neurons"] >= target_pop["start"]) & (network["target_neurons"] < target_pop["end"])
    return cp.where(source_mask & target_mask)[0]

def get_average_weight(network, indices):
    if len(indices) == 0:
        return 0.0
    return cp.mean(network["weights"][indices]).item()

def main():
    print("=== RUNNING HEADLESS INTUITIVE PHYSICS ENGINE (IPE) TEST ===")
    base_path = Path(__file__).resolve().parent.parent.parent
    network_config = load_config(base_path / "config" / "network" / "intuitive_physics_test.json")
    sim_config = load_config(base_path / "config" / "simulation" / "default_run.json")
    print("Building IPE network...")
    network, pop_info = build_network(network_config)
    simulator = Simulator(network, pop_info, sim_config)
    w1_indices = get_synapse_indices(network, pop_info, "Push_A_Motor", "A_is_Moving")
    w2_indices = get_synapse_indices(network, pop_info, "A_is_Moving", "B_is_Moving")
    w1_initial = get_average_weight(network, w1_indices)
    w2_initial = get_average_weight(network, w2_indices)
    print(f"\nInitial Avg. Weight (Push → A_is_Moving): {w1_initial:.4f}")
    print(f"Initial Avg. Weight (A_is_Moving → B_is_Moving): {w2_initial:.4f}\n")
    num_trials = 50
    trial_duration_ms = 100
    trial_steps = int(trial_duration_ms / sim_config['dt'])
    print(f"=== STARTING TRAINING: {num_trials} trials ===")
    for i in range(num_trials):
        if (i+1) % 10 == 0:
            print(f"Running Trial {i+1}/{num_trials}")
        for step in range(trial_steps):
            current_time_ms = step * sim_config['dt']
            spikes_to_inject = []
            if 0 <= current_time_ms < 10:   # push A
                pop = pop_info['Push_A_Motor']
                spikes_to_inject.append(cp.arange(pop['start'], pop['end'], dtype=cp.int32))
            if 15 <= current_time_ms < 35:  # A moves
                pop = pop_info['Object_A_Detector']
                spikes_to_inject.append(cp.arange(pop['start'], pop['end'], dtype=cp.int32))
            if 40 <= current_time_ms < 60:  # B moves
                pop = pop_info['Object_B_Detector']
                spikes_to_inject.append(cp.arange(pop['start'], pop['end'], dtype=cp.int32))
            sensory_spikes = cp.concatenate(spikes_to_inject) if spikes_to_inject else None
            simulator.step(sensory_spikes_indices=sensory_spikes)
    print("=== TRAINING COMPLETE ===\n")
    w1_final = get_average_weight(network, w1_indices)
    w2_final = get_average_weight(network, w2_indices)
    print("=== LEARNING VERIFICATION ===")
    print(f"Avg. Weight (Push → A_is_Moving): {w1_initial:.4f} → {w1_final:.4f}")
    print(f"Avg. Weight (A_is_Moving → B_is_Moving): {w2_initial:.4f} → {w2_final:.4f}\n")
    if w1_final > w1_initial * 1.5 and w2_final > w2_initial * 1.5:
        print("SUCCESS: Causal links were learned and strengthened via STDP.")
    else:
        print("FAILURE: Synaptic weights did not increase significantly. Learning did not occur.")

main()