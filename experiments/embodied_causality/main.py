import sys
from pathlib import Path
import numpy as np
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from isaacsim import SimulationApp
simulation_app_instance = SimulationApp({"headless": False})
from omni.isaac.core import World
from src.environment import table_scene

def run_simulation():
    world = World(stage_units_in_meters=1.0)
    print("Setting up scene...")
    robot_controller, ball_a, ball_b = table_scene.setup_scene(world)
    print("Resetting world to initialize physics...")
    world.reset()
    print("Initializing robot controller...")
    robot_controller.initialize()
    print("Starting simulation trials...")
    # sim loop
    num_trials = 10
    trial_duration_steps = 400
    push_duration_steps = 100
    retract_duration_steps = 100
    for i in range(num_trials):
        print(f"=== Running Trial {i+1}/{num_trials} ===")
        # reset to start.
        ball_a.set_world_pose(position=np.array([0, 0.4, 0.75]))
        ball_b.set_world_pose(position=np.array([0, 0.6, 0.75]))
        ball_a.set_linear_velocity(np.zeros(3))
        ball_b.set_linear_velocity(np.zeros(3))
        robot_controller.retract()
        for _ in range(20):
            world.step(render=True)
        # NOTE: this will execute one trial per frame update; for now it's fine.
        for step in range(trial_duration_steps):
            if step < push_duration_steps:
                # extend arm.
                robot_controller.push_forward()
            elif step < push_duration_steps + retract_duration_steps:
                # retract.
                robot_controller.retract()
            world.step(render=True)
        if not simulation_app_instance.is_running():
            break
    print("Simulation complete!")
    simulation_app_instance.close()

run_simulation()