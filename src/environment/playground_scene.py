import numpy as np
from omni.isaac.core.objects import DynamicSphere, FixedCuboid

def setup_scene(world):
    # simple ground plane
    world.scene.add_default_ground_plane()
    # some objects to interact with
    red_ball = world.scene.add(
        DynamicSphere(
            prim_path="/World/red_ball",
            name="red_ball",
            position=np.array([1.0, 0.0, 0.05]),
            radius=0.05,
            color=np.array([1.0, 0.0, 0.0])
        )
    )
    blue_cube = world.scene.add(
        FixedCuboid(
            prim_path="/World/blue_cube",
            name="blue_cube",
            position=np.array([-1.0, 0.5, 0.1]),
            scale=np.array([0.2, 0.2, 0.2]),
            color=np.array([0.0, 0.0, 1.0])
        )
    )
    return [red_ball, blue_cube]