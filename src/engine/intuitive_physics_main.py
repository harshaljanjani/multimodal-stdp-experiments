import sys
from pathlib import Path
import numpy as np
import cupy as cp
from datetime import datetime
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.loader import load_config
from src.network.builder import build_network
from src.engine.simulate import Simulator
from src.utils.analysis_utils import (
    get_synapse_indices, 
    get_average_weight, 
    get_weight_std,
    save_learning_plots,
    print_learning_results
)
try:
    from omni.isaac.kit import SimulationApp
except ImportError:
    print("Error: Isaac Sim libraries not found. Please run this script using the appropriate Isaac Sim launcher (e.g., ./python.sh).")
    sys.exit(1)

def encode_velocity_to_spikes(pop_info, pop_name, velocity_magnitude):
    if velocity_magnitude < 0.1:
        return None
    pop = pop_info[pop_name]
    num_firing = int(pop['count'] * min(0.5, velocity_magnitude / 50.0))
    if num_firing == 0:
        return None
    firing_indices = np.random.choice(
        np.arange(pop['start'], pop['end']), 
        size=num_firing, 
        replace=False
    )
    return cp.asarray(firing_indices, dtype=cp.int32)

def run_simulation(simulation_app):
    print("[DEBUG] run_simulation coroutine started.")
    from omni.isaac.core import World
    from omni.isaac.core.objects import DynamicSphere
    from pxr import Gf
    base_path = Path(__file__).resolve().parent.parent.parent
    network_config = load_config(base_path / "config" / "network" / "intuitive_physics_test.json")
    sim_config = load_config(base_path / "config" / "simulation" / "default_run.json")
    print("Building Intuitive Physics Engine (IPE) network...")
    network, pop_info = build_network(network_config)
    print(f"Network built with {network['membrane_potential'].shape[0]} neurons and {network['source_neurons'].shape[0]} synapses.")
    # SNN sim.
    print("[DEBUG] Creating SNN Simulator object...")
    snn_simulator = Simulator(network, pop_info, sim_config)
    print("[DEBUG] SNN Simulator object created.")
    push_to_moving_a_synapses = get_synapse_indices(network, pop_info, "Push_A_Motor", "A_is_Moving")
    moving_a_to_moving_b_synapses = get_synapse_indices(network, pop_info, "A_is_Moving", "B_is_Moving")
    weight_history = {
        'w1_mean': [],
        'w1_std': [],
        'w2_mean': [],
        'w2_std': [],
        'motor_spikes': [],
        'sensor_a_spikes': [],
        'sensor_b_spikes': []
    }
    w1_initial = get_average_weight(network, push_to_moving_a_synapses)
    w2_initial = get_average_weight(network, moving_a_to_moving_b_synapses)
    print(f"\nInitial Avg. Weight (Push → A_is_Moving): {w1_initial:.4f}")
    print(f"Initial Avg. Weight (A_is_Moving → B_is_Moving): {w2_initial:.4f}\n")
    # world setup.
    print("[DEBUG] Setting up Isaac Sim world...")
    world = World()
    world.scene.add_default_ground_plane()
    # create the billiard balls
    ball_a = world.scene.add(
        DynamicSphere(
            prim_path="/World/ball_a",
            name="ball_a",
            position=np.array([0, -0.5, 0.2]),
            radius=0.1,
            color=np.array([1.0, 0.0, 0.0])
        )
    )
    ball_b = world.scene.add(
        DynamicSphere(
            prim_path="/World/ball_b",
            name="ball_b",
            position=np.array([0, 0.5, 0.2]),
            radius=0.1,
            color=np.array([0.0, 0.0, 1.0])
        )
    )
    print("[DEBUG] Resetting world...")
    world.reset()
    print("[DEBUG] world.reset() has completed.")
    num_trials = 50
    # each trial is 200 ms.
    trial_duration_ms = 200
    trial_steps = int(trial_duration_ms / sim_config['dt'])
    sample_interval = 1
    print(f"=== STARTING TRAINING: {num_trials} trials ===")
    for i in range(num_trials):
        print(f"Running Trial {i+1}/{num_trials}")
        ball_a.set_world_pose(position=np.array([0, -0.5, 0.2]))
        ball_b.set_world_pose(position=np.array([0, 0.5, 0.2]))
        ball_a.set_linear_velocity(np.zeros(3))
        ball_b.set_linear_velocity(np.zeros(3))
        ball_a.set_angular_velocity(np.zeros(3))
        ball_b.set_angular_velocity(np.zeros(3))
        for _ in range(10): 
            world.step(render=True)
        # track spikes for this trial
        trial_motor_spikes = 0
        trial_sensor_a_spikes = 0
        trial_sensor_b_spikes = 0
        for step in range(trial_steps):
            print(f"\r    → Trial {i+1}, Step {step+1}/{trial_steps}", end="")
            current_time_ms = step * sim_config['dt']
            motor_spikes = None
            # action.
            if 0 <= current_time_ms < 10:
                impulse_velocity = np.array([0, 2.5, 0])
                ball_a.set_linear_velocity(impulse_velocity)
                pop = pop_info['Push_A_Motor']
                motor_spikes = cp.arange(pop['start'], pop['end'], dtype=cp.int32)
                trial_motor_spikes += len(motor_spikes)
            # sense.
            vel_a = ball_a.get_linear_velocity()
            vel_b = ball_b.get_linear_velocity()
            spikes_a = encode_velocity_to_spikes(pop_info, "Object_A_Detector", np.linalg.norm(vel_a))
            spikes_b = encode_velocity_to_spikes(pop_info, "Object_B_Detector", np.linalg.norm(vel_b))
            
            if spikes_a is not None:
                trial_sensor_a_spikes += len(spikes_a)
            if spikes_b is not None:
                trial_sensor_b_spikes += len(spikes_b)
            spike_lists = [s for s in [motor_spikes, spikes_a, spikes_b] if s is not None]
            all_sensory_spikes = cp.concatenate(spike_lists) if spike_lists else None
            # think.
            snn_simulator.step(all_sensory_spikes)
            # act.
            world.step(render=True)
        print()
            
        # sample weights after each trial
        if (i + 1) % sample_interval == 0:
            w1_mean = get_average_weight(network, push_to_moving_a_synapses)
            w1_std = get_weight_std(network, push_to_moving_a_synapses)
            w2_mean = get_average_weight(network, moving_a_to_moving_b_synapses)
            w2_std = get_weight_std(network, moving_a_to_moving_b_synapses)
            weight_history['w1_mean'].append(w1_mean)
            weight_history['w1_std'].append(w1_std)
            weight_history['w2_mean'].append(w2_mean)
            weight_history['w2_std'].append(w2_std)
            weight_history['motor_spikes'].append(trial_motor_spikes)
            weight_history['sensor_a_spikes'].append(trial_sensor_a_spikes)
            weight_history['sensor_b_spikes'].append(trial_sensor_b_spikes)
            print(f"    [DEBUG] W1: {w1_mean:.4f} (±{w1_std:.4f}), W2: {w2_mean:.4f} (±{w2_std:.4f})")
            print(f"    [DEBUG] Spikes - Motor: {trial_motor_spikes}, Sensor A: {trial_sensor_a_spikes}, Sensor B: {trial_sensor_b_spikes}")
    print("=== TRAINING COMPLETE ===")
    world.pause()
    w1_final = get_average_weight(network, push_to_moving_a_synapses)
    w2_final = get_average_weight(network, moving_a_to_moving_b_synapses)
    print_learning_results(
        w1_initial, w1_final, 
        w2_initial, w2_final,
        connection1_label="Push → A_is_Moving",
        connection2_label="A_is_Moving → B_is_Moving",
        threshold=2.0
    )
    output_dir = base_path / "results" / "visualizations"
    save_learning_plots(
        weight_history, 
        output_dir,
        experiment_name="intuitive_physics",
        connection1_label="Push → A_is_Moving",
        connection2_label="A_is_Moving → B_is_Moving"
    )

if __name__ == "__main__":
    simulation_app = SimulationApp({"headless": False})
    print("[DEBUG] SimulationApp object created. Waiting for Isaac Sim to initialize...")
    try:
        run_simulation(simulation_app)
    except Exception as e:
        print(f"An error occurred during simulation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[DEBUG] Simulation finished or error occurred. Closing SimulationApp.")
        simulation_app.close()
