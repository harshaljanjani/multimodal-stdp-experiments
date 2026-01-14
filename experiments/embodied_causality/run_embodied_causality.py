import sys
from pathlib import Path
import numpy as np
import cupy as cp
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
# snn.
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
# isaac-sim.
from isaacsim import SimulationApp
simulation_app_instance = SimulationApp({"headless": False})
from omni.isaac.core import World
from src.environment import table_scene
from src.cognition.object_tracker import ObjectTracker
from src.cognition.intuitive_physics_engine import IntuitivePhysicsEngine

# world state to sensory spikes.
def encode_velocity_to_spikes(pop_info, pop_name, velocity_magnitude, threshold=0.1, max_vel=2.0):
    if velocity_magnitude < threshold:
        return None
    pop = pop_info[pop_name]
    num_firing = int(pop['count'] * min(1.0, velocity_magnitude / max_vel))
    if num_firing == 0:
        return None
    firing_indices = np.random.choice(
        np.arange(pop['start'], pop['end']), 
        size=num_firing,
        replace=False
    )
    return cp.asarray(firing_indices, dtype=cp.int32)

def run_simulation():
    base_path = Path(__file__).resolve().parent.parent.parent
    network_config = load_config(base_path / "config" / "network" / "embodied_causality_v0.json")
    sim_config = load_config(base_path / "config" / "simulation" / "default_run.json")
    print("Building Embodied Causality network...")
    network, pop_info = build_network(network_config)
    print(f"Network built with {network['membrane_potential'].shape[0]} neurons and {network['source_neurons'].shape[0]} synapses.")
    snn_simulator = Simulator(network, pop_info, sim_config)
    object_tracker = ObjectTracker()
    ipe = IntuitivePhysicsEngine(network, pop_info)
    motor_to_b_synapses = get_synapse_indices(network, pop_info, "Franka_Extend_Motor", "B_is_Moving")
    b_to_a_synapses = get_synapse_indices(network, pop_info, "B_is_Moving", "A_is_Moving")
    weight_history = {
        'w1_mean': [],
        'w1_std': [],
        'w2_mean': [],
        'w2_std': [],
        'motor_spikes': [],
        'sensor_a_spikes': [],
        'sensor_b_spikes': []
    }
    w1_initial = get_average_weight(network, motor_to_b_synapses)
    w2_initial = get_average_weight(network, b_to_a_synapses)
    print(f"\n=== INITIAL SYNAPTIC WEIGHTS ===")
    print(f"Motor → B_is_Moving: {w1_initial:.4f}")
    print(f"B_is_Moving → A_is_Moving: {w2_initial:.4f}\n")
    world = World(stage_units_in_meters=1.0)
    print("Setting up scene...")
    robot_controller, ball_a, ball_b = table_scene.setup_scene(world)
    for _ in range(4):
        simulation_app_instance.update()
    print("Resetting world to initialize physics...")
    world.reset()
    print("Initializing robot controller...")
    robot_controller.initialize()
    print("\n=== STARTING EMBODIED CAUSALITY TRAINING ===")
    # sim loop
    num_trials = 50
    trial_duration_steps = 200
    push_duration_steps = 80
    sample_interval = 1
    for i in range(num_trials):
        print(f"\n=== Running Trial {i+1}/{num_trials} ===")
        # reset to start.
        ball_a.set_world_pose(position=np.array([0, 0.4, 0.75]))
        ball_b.set_world_pose(position=np.array([0, 0.6, 0.75]))
        ball_a.set_linear_velocity(np.zeros(3))
        ball_b.set_linear_velocity(np.zeros(3))
        ball_a.set_angular_velocity(np.zeros(3))
        ball_b.set_angular_velocity(np.zeros(3))
        robot_controller.retract()
        for _ in range(20):
            simulation_app_instance.update()
        trial_motor_spikes = 0
        trial_sensor_a_spikes = 0
        trial_sensor_b_spikes = 0
        for step in range(trial_duration_steps):
            # sense.
            vel_a = ball_a.get_linear_velocity()
            vel_b = ball_b.get_linear_velocity()
            spikes_b = encode_velocity_to_spikes(pop_info, "Object_B_Detector", np.linalg.norm(vel_a))
            spikes_a = encode_velocity_to_spikes(pop_info, "Object_A_Detector", np.linalg.norm(vel_b))
            if spikes_a is not None:
                trial_sensor_a_spikes += len(spikes_a)
            if spikes_b is not None:
                trial_sensor_b_spikes += len(spikes_b)
            # think.
            motor_spikes = None
            if step < push_duration_steps:
                # extend arm.
                robot_controller.push_forward()
                # this is our scripted "intent to push"; we inject these spikes into the SNN for now.
                pop = pop_info['Franka_Extend_Motor']
                motor_spikes = cp.arange(pop['start'], pop['end'], dtype=cp.int32)
                trial_motor_spikes += len(motor_spikes)
            else:
                # retract.
                robot_controller.retract()
            # act -> handled by `robot_controller` calls.
            spike_lists = [s for s in [motor_spikes, spikes_a, spikes_b] if s is not None]
            all_spikes = cp.concatenate(spike_lists) if spike_lists else None
            snn_simulator.step(sensory_spikes_indices=all_spikes)
            simulation_app_instance.update()
            if step % 10 == 0:
                # we don't have camera here, so we'll manually register balls
                ball_a_pos, _ = ball_a.get_world_pose()
                ball_b_pos, _ = ball_b.get_world_pose()
                # fake "vision" by directly registering object positions
                object_tracker.objects[0] = {
                    "position": ball_a_pos[:2],
                    "velocity": ball_a.get_linear_velocity()[:2],
                    "color": "blue",
                    "last_seen": step * sim_config["dt"],
                    "created_at": 0
                }
                object_tracker.objects[1] = {
                    "position": ball_b_pos[:2],
                    "velocity": ball_b.get_linear_velocity()[:2],
                    "color": "red",
                    "last_seen": step * sim_config["dt"],
                    "created_at": 0
                }
                ipe.extract_beliefs_from_snn(object_tracker)
        if (i + 1) % 10 == 0:
            ipe.print_beliefs()
        if (i + 1) % sample_interval == 0:
            w1_mean = get_average_weight(network, motor_to_b_synapses)
            w1_std = get_weight_std(network, motor_to_b_synapses)
            w2_mean = get_average_weight(network, b_to_a_synapses)
            w2_std = get_weight_std(network, b_to_a_synapses)
            weight_history['w1_mean'].append(w1_mean)
            weight_history['w1_std'].append(w1_std)
            weight_history['w2_mean'].append(w2_mean)
            weight_history['w2_std'].append(w2_std)
            weight_history['motor_spikes'].append(trial_motor_spikes)
            weight_history['sensor_a_spikes'].append(trial_sensor_a_spikes)
            weight_history['sensor_b_spikes'].append(trial_sensor_b_spikes)
            print(f"  [WEIGHTS] Motor→B: {w1_mean:.4f} (±{w1_std:.4f}), B→A: {w2_mean:.4f} (±{w2_std:.4f})")
            print(f"  [SPIKES] Motor: {trial_motor_spikes}, Sensor A: {trial_sensor_a_spikes}, Sensor B: {trial_sensor_b_spikes}")
        if not simulation_app_instance.is_running():
            break
    print("\n=== TRAINING COMPLETE ===")
    w1_final = get_average_weight(network, motor_to_b_synapses)
    w2_final = get_average_weight(network, b_to_a_synapses)
    print_learning_results(
        w1_initial, w1_final, w2_initial, w2_final,
        "Motor → B_is_Moving", "B_is_Moving → A_is_Moving",
        threshold=1.5
    )
    output_dir = base_path / "results" / "visualizations" / "embodied_causality"
    save_learning_plots(
        weight_history, 
        output_dir,
        experiment_name="embodied_causality_learning",
        connection1_label="Motor → B_is_Moving",
        connection2_label="B_is_Moving → A_is_Moving"
    )
    print("\nSimulation complete!")
    simulation_app_instance.close()

run_simulation()