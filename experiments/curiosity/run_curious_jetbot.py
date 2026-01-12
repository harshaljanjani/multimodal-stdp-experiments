import sys
from pathlib import Path
import numpy as np
import cupy as cp
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
# isaac-sim
from isaacsim import SimulationApp
simulation_app_instance = SimulationApp({"headless": False})
from omni.isaac.core import World
from src.environment import playground_scene
from src.agent.jetbot_controller import JetbotController
from src.perception.vision import VisionSystem
from src.perception import spike_encoder
from src.cognition.curiosity_engine import CuriosityEngine
from src.utils.loader import load_config
from src.network.builder import build_network
from src.engine.simulate import Simulator

def run_simulation():
    base_path = Path(__file__).resolve().parent.parent.parent
    network_config = load_config(base_path / "config" / "network" / "curious_jetbot_v0.json")
    sim_config = load_config(base_path / "config" / "simulation" / "default_run.json")
    world = World(stage_units_in_meters=1.0)
    # environment
    playground_objects = playground_scene.setup_scene(world)
    # agent
    robot_controller = JetbotController()
    robot = robot_controller.create_robot(world)
    world.reset()
    robot_controller.initialize()
    camera_path = "/World/JetbotCamera"
    # perception
    vision = VisionSystem(
        camera_prim_path=camera_path,
        attachment_prim_path="/World/Jetbot/chassis",
        offset_position=[0.1, 0.0, 0.05]
    )
    vision.initialize(world)
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
    print("\n=== STARTING CURIOUS EXPLORATION ===")
    max_steps = 5000
    for step in range(max_steps):
        vision.update_camera_pose()
        # sense.
        rgba_data = vision.camera.get_rgba()
        img_gpu = cp.asarray(rgba_data[..., :3]) if rgba_data is not None else None
        sensory_spikes = spike_encoder.encode_spatial_location(
            img_gpu, pop_info, vision_pop_map, target_color_tensor
        )
        # think.
        snn_simulator.step(sensory_spikes)
        motor_rates = snn_simulator.get_motor_firing_rates()
        # decide (via curiosity).
        action = curiosity_engine.step(img_gpu, motor_rates)
        # act.
        if action == "forward":
            robot_controller.forward()
        elif action == "turn_left":
            robot_controller.turn_left()
        elif action == "turn_right":
            robot_controller.turn_right()
        else: # stop
            robot_controller.stop()
        # update isaac sim
        world.step(render=True)
        if step % 100 == 0:
            print(f"Step {step}/{max_steps} | Action: {action} | Motor Rates: {motor_rates}")
    print("\n=== EXPLORATION COMPLETE ===")
    simulation_app_instance.close()

if __name__ == "__main__":
    run_simulation()