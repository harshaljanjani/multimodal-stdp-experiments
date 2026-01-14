import numpy as np
from omni.isaac.core.objects import DynamicSphere, FixedCuboid
from src.agent.franka_controller import FrankaController

def setup_scene(world):
    # add table.
    world.scene.add_default_ground_plane()
    table = world.scene.add(
        FixedCuboid(
            prim_path="/World/table",
            name="table",
            position=np.array([0, 0, 0.35]),
            scale=np.array([0.8, 1.5, 0.7]),
            color=np.array([0.5, 0.5, 0.5]),
        )
    )
    base_rotation = np.pi / 2
    home_joints = np.array([base_rotation, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04])
    pre_push_joints = np.array([base_rotation, 0.95, 0.0, -1.8, 0.0, 2.1, 0.785, 0.04, 0.04])
    push_joints = np.array([base_rotation, 0.95, 0.0, -1.5, 0.0, 2.1, 0.785, 0.04, 0.04])
    robot_controller = FrankaController(
        prim_path="/World/Franka",
        name="franka_robot",
        position=np.array([0.0, 0.0, 0.7]),
        home_joint_positions=home_joints,
        pre_push_joint_positions=pre_push_joints,
        push_joint_positions=push_joints
    )
    robot = robot_controller.create_robot()
    world.scene.add(robot)
    # add objs.
    ball_a = world.scene.add(
        DynamicSphere(
            prim_path="/World/ball_a",
            name="ball_a",
            position=np.array([0, 0.4, 0.75]),
            radius=0.04,
            color=np.array([0.0, 0.0, 1.0])
        )
    )
    ball_b = world.scene.add(
        DynamicSphere(
            prim_path="/World/ball_b",
            name="ball_b",
            position=np.array([0, 0.6, 0.75]),
            radius=0.04,
            color=np.array([1.0, 0.0, 0.0])
        )
    )
    return robot_controller, ball_a, ball_b