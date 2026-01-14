import sys
from pathlib import Path
import numpy as np
import cupy as cp
import cv2
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
# isaac-sim
from isaacsim import SimulationApp
simulation_app_instance = SimulationApp({"headless": False})
from omni.isaac.core import World
from src.environment import rich_playground_scene
from src.agent.jetbot_controller import JetbotController
from src.perception.vision import VisionSystem
from src.perception.touch import TouchSystem
from src.perception.proprioception import ProprioceptionSystem
from src.perception import spike_encoder
from src.cognition.curiosity_engine import CuriosityEngine
from src.utils.loader import load_config
from src.network.builder import build_network
from src.engine.simulate import Simulator
from src.utils.analysis_utils import (
    get_average_weight,
    save_curiosity_plots,
    get_synapse_indices
)

def run_simulation():
    base_path = Path(__file__).resolve().parent.parent.parent
    network_config = load_config(base_path / "config" / "network" / "curious_jetbot_v0.json")
    sim_config = load_config(base_path / "config" / "simulation" / "default_run.json")
    world = World(stage_units_in_meters=1.0)
    # environment
    playground_objects = rich_playground_scene.setup_scene(world)
    # agent
    robot_controller = JetbotController()
    robot = robot_controller.create_robot(world)
    world.reset()
    robot_controller.initialize()
    robot_controller.forward_velocity = 25.0
    robot_controller.turn_velocity = 15.0
    camera_path = "/World/JetbotCamera"
    # perception
    vision = VisionSystem(
        camera_prim_path=camera_path,
        attachment_prim_path="/World/Jetbot/chassis",
        offset_position=[0.1, 0.0, 0.01]
    )
    vision.initialize(world)
    touch = TouchSystem(robot_prim_path="/World/Jetbot")
    touch.initialize()
    proprio = ProprioceptionSystem(robot_articulation=robot)
    proprio.initialize()
    # brain
    network, pop_info = build_network(network_config)
    snn_simulator = Simulator(network, pop_info, sim_config)
    action_space = ["forward", "turn_left", "turn_right", "stop"]
    curiosity_engine = CuriosityEngine(action_space)
    # map encoder regions to network population names
    vision_pop_map = {
        "Vision_Left": "Vision_Left",
        "Vision_Center": "Vision_Center",
        "Vision_Right": "Vision_Right"
    }
    target_color_tensor = cp.array([1.0, 0.0, 0.0], dtype=cp.float32)
    connections_to_monitor = {
        "VisionC_to_Integration": ("Vision_Center", "Sensory_Integration"),
        "Integration_to_MotorF": ("Sensory_Integration", "Motor_Forward"),
    }
    synapse_indices = {
        name: get_synapse_indices(network, pop_info, src, tgt)
        for name, (src, tgt) in connections_to_monitor.items()
    }
    populations_to_track_spikes = [
        "Vision_Center", "Sensory_Integration", "Motor_Forward", "Motor_Turn_L"
    ]
    spike_history_accumulator = {name: 0 for name in populations_to_track_spikes}
    learning_history = {
        'weights': {name: [] for name in connections_to_monitor},
        'spikes': {name: [] for name in populations_to_track_spikes}
    }
    debug_dir = Path("_debug_vision")
    debug_dir.mkdir(exist_ok=True)
    print(f"[DEBUG] Jetbot vision frames will be saved to: {debug_dir.resolve()}")
    print("\n=== STARTING CURIOUS EXPLORATION ===")
    max_steps = 5000
    sample_interval = 100
    for step in range(max_steps):
        vision.update_camera_pose()
        # sense.
        rgba_data = vision.camera.get_rgba()
        if rgba_data is None or rgba_data.shape[0] == 0:
            world.step(render=True)
            continue
        img_gpu = cp.asarray(rgba_data[..., :3])
        if step % 50 == 0:
            img_bgr = cv2.cvtColor(rgba_data, cv2.COLOR_RGBA2BGRA)
            filepath = str(debug_dir / f"jetbot_frame_{step:05d}.png")
            cv2.imwrite(filepath, img_bgr)
        vision_spikes = spike_encoder.encode_spatial_location(
            img_gpu, pop_info, vision_pop_map, target_color_tensor
        )
        # sense - touch
        contact_count = touch.get_contact_count()
        touch_spikes = touch.encode_touch_to_spikes(pop_info, "Touch_Sensor", contact_count)
        # sense - proprioception
        motion_intensity = proprio.get_motion_intensity()
        proprio_spikes = proprio.encode_motion_to_spikes(pop_info, "Proprio_Motion", motion_intensity)
        # combine all sensory spikes
        spike_lists = [s for s in [vision_spikes, touch_spikes, proprio_spikes] if s is not None]
        sensory_spikes = cp.concatenate(spike_lists) if spike_lists else None
        # think.
        spiked_this_step = snn_simulator.step(sensory_spikes)
        motor_rates = snn_simulator.get_motor_firing_rates()
        # decide (via curiosity).
        robot_pos, _ = robot.get_world_pose()
        action = curiosity_engine.step(img_gpu, robot_pos, motion_intensity)
        # act.
        if action == "forward":
            robot_controller.forward()
            for _ in range(3):
                world.step(render=True)
        elif action == "turn_left":
            robot_controller.turn_left()
            for _ in range(4):
                world.step(render=True)
            robot_controller.stop()
            world.step(render=True)
        elif action == "turn_right":
            robot_controller.turn_right()
            for _ in range(4):
                world.step(render=True)
            robot_controller.stop()
            world.step(render=True)
        else: # stop
            robot_controller.stop()
            # update isaac sim
            world.step(render=True)
        for pop_name in populations_to_track_spikes:
            pop = pop_info[pop_name]
            spike_history_accumulator[pop_name] += int(cp.sum(spiked_this_step[pop['start']:pop['end']]).item())
        if step > 0 and step % sample_interval == 0:
            for name, indices in synapse_indices.items():
                avg_w = get_average_weight(network, indices)
                learning_history['weights'][name].append(avg_w)
            for name, total_spikes in spike_history_accumulator.items():
                learning_history['spikes'][name].append(total_spikes)
                spike_history_accumulator[name] = 0
            avg_pred_error = np.mean(curiosity_engine.prediction_errors[-10:]) if curiosity_engine.prediction_errors else 0.0
            stuck_status = "RECOVERING" if curiosity_engine.recovery_mode else "exploring"
            print(f"Step {step}/{max_steps} | Action: {action} | Status: {stuck_status} | Motor Rates: {motor_rates} | Pred Error: {avg_pred_error:.3f}")
    print("\n=== EXPLORATION COMPLETE ===")
    output_dir = base_path / "results" / "visualizations" / "curious_jetbot"
    save_curiosity_plots(learning_history, output_dir)
    simulation_app_instance.close()

if __name__ == "__main__":
    run_simulation()