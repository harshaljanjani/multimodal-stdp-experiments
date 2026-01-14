# scripts/experiments/run_goal_directed_learning.py
import sys
from pathlib import Path
import cupy as cp
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from isaacsim import SimulationApp
simulation_app_instance = SimulationApp({"headless": False})
from omni.isaac.core import World
from src.environment import rich_playground_scene
from src.agent.jetbot_controller import JetbotController
from src.perception.vision import VisionSystem
from src.perception.touch import TouchSystem
from src.perception.proprioception import ProprioceptionSystem
from src.cognition.object_tracker import ObjectTracker
from src.cognition.intuitive_physics_engine import IntuitivePhysicsEngine
from src.cognition.goal_generator import GoalGenerator
from src.cognition.curiosity_engine import CuriosityEngine
from src.utils.loader import load_config
from src.network.builder import build_network
from src.engine.simulate import Simulator

def run_simulation():
    base_path = Path(__file__).resolve().parent.parent.parent
    network_config = load_config(base_path / "config" / "network" / "curious_jetbot_v0.json")
    sim_config = load_config(base_path / "config" / "simulation" / "default_run.json")
    world = World(stage_units_in_meters=1.0)
    playground_objects = rich_playground_scene.setup_scene(world)
    robot_controller = JetbotController()
    robot = robot_controller.create_robot(world)
    world.reset()
    robot_controller.initialize()
    robot_controller.forward_velocity = 20.0
    robot_controller.turn_velocity = 12.0
    camera_path = "/World/JetbotCamera"
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
    network, pop_info = build_network(network_config)
    snn_simulator = Simulator(network, pop_info, sim_config)
    action_space = ["forward", "turn_left", "turn_right", "stop"]
    curiosity_engine = CuriosityEngine(action_space)
    object_tracker = ObjectTracker()
    ipe = IntuitivePhysicsEngine(network, pop_info)
    goal_generator = GoalGenerator(action_space)
    print("\n=== STARTING GOAL-DIRECTED LEARNING ===")
    max_steps = 3000
    report_interval = 100
    goal_duration = 200
    steps_in_current_goal = 0
    for step in range(max_steps):
        current_time_ms = step * sim_config["dt"]
        vision.update_camera_pose()
        rgba_data = vision.camera.get_rgba()
        if rgba_data is None or rgba_data.shape[0] == 0:
            world.step(render=True)
            continue
        img_gpu = cp.asarray(rgba_data[..., :3])
        # update cognitive modules
        object_tracker.update_from_vision(img_gpu, current_time_ms)
        ipe.extract_beliefs_from_snn(object_tracker)
        # generate/update goal
        if steps_in_current_goal == 0 or steps_in_current_goal >= goal_duration:
            goal = goal_generator.generate_goal(object_tracker, ipe, curiosity_engine)
            goal_generator.print_goal(goal)
            steps_in_current_goal = 0
        steps_in_current_goal += 1
        # sense
        contact_count = touch.get_contact_count()
        touch_spikes = touch.encode_touch_to_spikes(pop_info, "Touch_Sensor", contact_count)
        motion_intensity = proprio.get_motion_intensity()
        proprio_spikes = proprio.encode_motion_to_spikes(pop_info, "Proprio_Motion", motion_intensity)
        # for now we still use curiosity engine for action selection
        # later: goal_generator will directly control actions
        spike_lists = [s for s in [touch_spikes, proprio_spikes] if s is not None]
        sensory_spikes = cp.concatenate(spike_lists) if spike_lists else None
        # think
        spiked_this_step = snn_simulator.step(sensory_spikes)
        motor_rates = snn_simulator.get_motor_firing_rates()
        # decide
        robot_pos, _ = robot.get_world_pose()
        action = curiosity_engine.step(img_gpu, robot_pos, motion_intensity)
        # act
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
        else:
            robot_controller.stop()
            world.step(render=True)
        # periodic reporting
        if step > 0 and step % report_interval == 0:
            print(f"\n=== Step {step}/{max_steps} ===")
            object_tracker.print_status()
            ipe.print_beliefs()
            goal_generator.print_goal(goal_generator.get_active_goal())
    print("\n=== GOAL-DIRECTED LEARNING COMPLETE ===")
    simulation_app_instance.close()

if __name__ == "__main__":
    run_simulation()