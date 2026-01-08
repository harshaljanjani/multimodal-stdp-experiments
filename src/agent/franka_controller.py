import numpy as np
from omni.isaac.core.utils.types import ArticulationAction

class FrankaController:
    def __init__(self, prim_path="/World/Franka", name="franka_robot", position=np.array([0.0, 0.0, 0.0])):
        self.prim_path = prim_path
        self.name = name
        self.position = position
        base_rotation = np.pi / 2
        self.home_joint_positions = np.array([base_rotation, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04])
        self.pre_push_joint_positions = np.array([base_rotation, 0.95, 0.0, -1.8, 0.0, 2.1, 0.785, 0.04, 0.04])
        self.push_joint_positions = np.array([base_rotation, 0.95, 0.0, -1.5, 0.0, 2.1, 0.785, 0.04, 0.04])
        self._robot_articulation = None
        self._articulation_controller = None

    def create_robot(self):
        from omni.isaac.franka import Franka
        self._robot_articulation = Franka(
            prim_path=self.prim_path,
            name=self.name,
            position=self.position
        )
        print(f"Created Franka robot at {self.prim_path}")
        return self._robot_articulation

    def initialize(self):
        print("Initializing robot controller...")
        if self._articulation_controller is None:
            self._articulation_controller = self._robot_articulation.get_articulation_controller()
        self.retract()
        print("Robot controller initialized")

    def go_to_pre_push(self):
        if self._articulation_controller is not None:
            action = ArticulationAction(joint_positions=self.pre_push_joint_positions)
            self._articulation_controller.apply_action(action)

    def push_forward(self):
        # push pos.
        if self._articulation_controller is not None:
            action = ArticulationAction(joint_positions=self.push_joint_positions)
            self._articulation_controller.apply_action(action)
        
    def retract(self):
        # home / retracted pos.
        if self._articulation_controller is not None:
            action = ArticulationAction(joint_positions=self.home_joint_positions)
            self._articulation_controller.apply_action(action)
