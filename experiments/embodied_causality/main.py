import sys
from pathlib import Path
import numpy as np
import cupy as cp
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
# snn.
from src.utils.loader import load_config
from src.network.builder import build_network
from src.engine.simulate import Simulator
# isaac-sim.
from isaacsim import SimulationApp
simulation_app_instance = SimulationApp({"headless": False})
from omni.isaac.core import World
from src.environment import table_scene

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
    network, pop_info = build_network(network_config)
    snn_simulator = Simulator(network, pop_info, sim_config)
    world = World(stage_units_in_meters=1.0)
    print("Setting up scene...")
    robot_controller, ball_a, ball_b = table_scene.setup_scene(world)
    for _ in range(4):
        simulation_app_instance.update()
    print("Resetting world to initialize physics...")
    world.reset()
    print("Initializing robot controller...")
    robot_controller.initialize()
    print("Starting simulation trials...")
    # sim loop
    num_trials = 50
    trial_duration_steps = 200
    push_duration_steps = 80
    for i in range(num_trials):
        print(f"=== Running Trial {i+1}/{num_trials} ===")
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
        for step in range(trial_duration_steps):
            # sense.
            vel_a = ball_a.get_linear_velocity()
            vel_b = ball_b.get_linear_velocity()
            spikes_a = encode_velocity_to_spikes(pop_info, "Object_A_Detector", np.linalg.norm(vel_a))
            spikes_b = encode_velocity_to_spikes(pop_info, "Object_B_Detector", np.linalg.norm(vel_b))
            # think.
            motor_spikes = None
            if step < push_duration_steps:
                # extend arm.
                robot_controller.push_forward()
                # this is our scripted "intent to push"; we inject these spikes into the SNN for now.
                pop = pop_info['Franka_Extend_Motor']
                motor_spikes = cp.arange(pop['start'], pop['end'], dtype=cp.int32)
            else:
                # retract.
                robot_controller.retract()
            # act -> handled by `robot_controller` calls.
            spike_lists = [s for s in [motor_spikes, spikes_a, spikes_b] if s is not None]
            all_spikes = cp.concatenate(spike_lists) if spike_lists else None
            snn_simulator.step(sensory_spikes_indices=all_spikes)
            simulation_app_instance.update()
        if not simulation_app_instance.is_running():
            break
    print("Simulation complete!")
    simulation_app_instance.close()

run_simulation()