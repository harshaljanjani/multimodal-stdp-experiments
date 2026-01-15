import sys
from pathlib import Path
import numpy as np
import cupy as cp
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from isaacsim import SimulationApp
simulation_app_instance = SimulationApp({"headless": False})
from omni.isaac.core import World
# project imports
from src.utils.loader import load_config
from src.network.builder import build_network
from src.engine.simulate import Simulator
from src.agent.embodied_agent import EmbodiedAgent
from src.cognition.causal_memory import CausalMemory
from src.environment import tool_use_scene
from src.utils.analysis_utils import get_synapse_indices, get_average_weight

def run_simulation():
    base_path = Path(__file__).resolve().parent.parent.parent
    network_config = load_config(base_path / "config" / "network" / "tool_use_v0.json")
    sim_config = load_config(base_path / "config" / "simulation" / "default_run.json")
    # setup world and robot
    world = World(stage_units_in_meters=1.0)
    robot_controller, stick, ball = tool_use_scene.setup_scene(world)
    world.reset()
    robot_controller.initialize()
    # build brain
    network, pop_info = build_network(network_config)
    snn_simulator = Simulator(network, pop_info, sim_config)
    # build agent
    agent = EmbodiedAgent(robot_controller, None, snn_simulator, pop_info)
    causal_mem = CausalMemory()
    # get synapse indices to monitor
    ie_to_ie_synapses = get_synapse_indices(network, pop_info, "Internal_E", "Internal_E")
    ie_to_motor_synapses = get_synapse_indices(network, pop_info, "Internal_E", "Motor_Push_Stick")
    w1_initial = get_average_weight(network, ie_to_ie_synapses)
    w2_initial = get_average_weight(network, ie_to_motor_synapses)
    print(f"\nInitial Avg. Weight (Internal→Internal): {w1_initial:.4f}")
    print(f"Initial Avg. Weight (Internal→Motor): {w2_initial:.4f}\n")
    # simulation loop
    num_trials = 50
    trial_duration_steps = 300
    action_duration_steps = 150
    stick_reset_pos, _ = stick.get_world_pose()
    ball_reset_pos, _ = ball.get_world_pose()
    default_orientation = np.array([1.0, 0.0, 0.0, 0.0])
    print(f"=== STARTING TOOL USE TRAINING: {num_trials} trials ===")
    for i in range(num_trials):
        print(f"\n=== Running Trial {i+1}/{num_trials} ===")
        # reset scene
        robot_controller.retract()
        stick.set_world_pose(position=stick_reset_pos, orientation=default_orientation)
        stick.set_linear_velocity(np.zeros(3))
        stick.set_angular_velocity(np.zeros(3))
        ball.set_world_pose(position=ball_reset_pos, orientation=default_orientation)
        ball.set_linear_velocity(np.zeros(3))
        ball.set_angular_velocity(np.zeros(3))
        for _ in range(20):
            simulation_app_instance.update()
        for step in range(trial_duration_steps):
            command_spikes = None
            # scripted action: push the stick
            if step < 50:
                robot_controller.go_to_pre_push() # a bit of a hack to aim it
            elif step < action_duration_steps:
                robot_controller.push_forward()
                # inject "intent to act" spikes into `Command_Go`
                pop = pop_info['Command_Go']
                command_spikes = cp.arange(pop['start'], pop['end'], dtype=cp.int32)
            else:
                robot_controller.retract()
            # agent performs a cycle
            agent.step(stick, ball, motor_command_spikes=command_spikes)
            # update simulation
            simulation_app_instance.update()
        w1_current = get_average_weight(network, ie_to_ie_synapses)
        w2_current = get_average_weight(network, ie_to_motor_synapses)
        print(f"  Trial Complete. W1: {w1_current:.4f}, W2: {w2_current:.4f}")
    print("\n=== TRAINING COMPLETE ===")
    w1_final = get_average_weight(network, ie_to_ie_synapses)
    w2_final = get_average_weight(network, ie_to_motor_synapses)
    print("\n=== FINAL WEIGHTS ===")
    print(f"Avg. Weight (Internal→Internal): {w1_initial:.4f} → {w1_final:.4f}")
    print(f"Avg. Weight (Internal→Motor): {w2_initial:.4f} → {w2_final:.4f}\n")
    # a simple check to see if any learning happened
    if w1_final > w1_initial * 1.5 and w2_final > w2_initial * 1.5:
        print("✓ SUCCESS: Hebbian learning strengthened internal and motor pathways.")
        causal_mem.add_or_strengthen_link("pull_stick", "move_ball", w1_final)
    else:
        print("✗ FAILURE: Weights did not increase significantly.")
    causal_mem.print_memory()
    simulation_app_instance.close()

if __name__ == "__main__":
    run_simulation()