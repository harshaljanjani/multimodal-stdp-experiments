import numpy as np
from omni.isaac.core.utils.types import ArticulationAction

class FrankaController:
    def __init__(self, prim_path="/World/Franka", name="franka_robot", position=np.array([0.0, 0.0, 0.0]), home_joint_positions=None, pre_push_joint_positions=None, push_joint_positions=None):
        self.prim_path = prim_path
        self.name = name
        self.position = position
        self.home_joint_positions = home_joint_positions
        self.pre_push_joint_positions = pre_push_joint_positions
        self.push_joint_positions = push_joint_positions
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
        if self._articulation_controller is not None and self.pre_push_joint_positions is not None:
            action = ArticulationAction(joint_positions=self.pre_push_joint_positions)
            self._articulation_controller.apply_action(action)

    def push_forward(self):
        # push pos.
        if self._articulation_controller is not None and self.push_joint_positions is not None:
            action = ArticulationAction(joint_positions=self.push_joint_positions)
            self._articulation_controller.apply_action(action)
        
    def retract(self):
        # home / retracted pos.
        if self._articulation_controller is not None and self.home_joint_positions is not None:
            action = ArticulationAction(joint_positions=self.home_joint_positions)
            self._articulation_controller.apply_action(action)
