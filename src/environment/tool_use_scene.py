import numpy as np
from omni.isaac.core.objects import DynamicSphere, DynamicCuboid, FixedCuboid
from src.agent.franka_controller import FrankaController

def setup_scene(world):
    world.scene.add_default_ground_plane()
    table_surface_z = 0.7
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
    pre_push_joints = np.array([base_rotation, 1.1, 0.0, -1.6, 0.0, 1.9, 0.785, 0.04, 0.04])
    push_joints = np.array([base_rotation, 0.8, 0.0, -1.9, 0.0, 2.2, 0.785, 0.04, 0.04])
    robot_controller = FrankaController(
        prim_path="/World/Franka",
        name="franka_robot",
        position=np.array([0.0, 0.0, table_surface_z]),
        home_joint_positions=home_joints,
        pre_push_joint_positions=pre_push_joints,
        push_joint_positions=push_joints
    )
    robot = robot_controller.create_robot()
    world.scene.add(robot)
    ball_radius = 0.03
    ball = world.scene.add(
        DynamicSphere(
            prim_path="/World/target_ball",
            name="target_ball",
            position=np.array([0.0, 0.4, table_surface_z + ball_radius]),
            radius=ball_radius,
            color=np.array([0.0, 0.0, 1.0]) # blue
        )
    )
    stick_half_height = 0.01
    stick = world.scene.add(
        DynamicCuboid(
            prim_path="/World/tool_stick",
            name="tool_stick",
            position=np.array([0.0, 0.6, table_surface_z + stick_half_height]),
            scale=np.array([0.1, 0.3, 0.02]),
            color=np.array([1.0, 0.0, 0.0]) # red
        )
    )
    return robot_controller, stick, ball