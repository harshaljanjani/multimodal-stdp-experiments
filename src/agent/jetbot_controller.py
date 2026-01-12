import numpy as np
from omni.isaac.core.utils.types import ArticulationAction

class JetbotController:
    def __init__(self, prim_path="/World/Jetbot", name="jetbot", position=np.array([0.0, 0.0, 0.03])):
        self.prim_path = prim_path
        self.name = name
        self.position = position
        self._robot_articulation = None
        self.forward_velocity = 10.0
        self.turn_velocity = 5.0

    def create_robot(self, world):
        from omni.isaac.wheeled_robots.robots import WheeledRobot
        self._robot_articulation = WheeledRobot(
            prim_path=self.prim_path,
            name=self.name,
            wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
            create_robot=True,
            usd_path=r"https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd",
            position=self.position
        )
        world.scene.add(self._robot_articulation)
        print(f"Created Jetbot at {self.prim_path}")
        return self._robot_articulation
    
    def initialize(self):
        if self._robot_articulation:
            self._robot_articulation.initialize()

    def forward(self):
        velocities = np.array([self.forward_velocity, self.forward_velocity])
        self.set_wheel_velocity(velocities)

    def turn_left(self):
        velocities = np.array([-self.turn_velocity, self.turn_velocity])
        self.set_wheel_velocity(velocities)

    def turn_right(self):
        velocities = np.array([self.turn_velocity, -self.turn_velocity])
        self.set_wheel_velocity(velocities)

    def stop(self):
        velocities = np.array([0.0, 0.0])
        self.set_wheel_velocity(velocities)
    
    def set_wheel_velocity(self, velocities: np.ndarray):
        if self._robot_articulation:
            action = ArticulationAction(joint_velocities=velocities)
            self._robot_articulation.apply_wheel_actions(action)

    def get_velocities(self):
        if self._robot_articulation:
            lin_vel = self._robot_articulation.get_linear_velocity()
            ang_vel = self._robot_articulation.get_angular_velocity()
            return lin_vel, ang_vel
        return None, None