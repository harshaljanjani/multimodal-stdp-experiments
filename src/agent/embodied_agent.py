import cupy as cp
import numpy as np

class EmbodiedAgent:
    def __init__(self, robot_controller, vision_system, snn_simulator, pop_info):
        self.robot_controller = robot_controller
        self.vision = vision_system
        self.snn = snn_simulator
        self.pop_info = pop_info
        print("[AGENT] EmbodiedAgent created and initialized.")

    def sense(self, stick_prim, ball_prim):
        # for now, we'll cheat a bit and use object prims to get velocities
        # instead of inferring from vision. this isolates the causality learning.
        stick_vel = np.linalg.norm(stick_prim.get_linear_velocity())
        ball_vel = np.linalg.norm(ball_prim.get_linear_velocity())
        spikes_stick = self._encode_velocity_to_spikes("Vision_Stick", stick_vel)
        spikes_ball = self._encode_velocity_to_spikes("Vision_Ball", ball_vel)
        spike_lists = [s for s in [spikes_stick, spikes_ball] if s is not None]
        return cp.concatenate(spike_lists) if spike_lists else None

    def _encode_velocity_to_spikes(self, pop_name, velocity, threshold=0.05, max_vel=1.0):
        if velocity < threshold:
            return None
        pop = self.pop_info[pop_name]
        num_firing = int(pop['count'] * min(1.0, velocity / max_vel))
        if num_firing == 0:
            return None
        return cp.random.choice(
            cp.arange(pop['start'], pop['end']),
            size=num_firing,
            replace=False
        ).astype(cp.int32)

    def think(self, sensory_spikes, motor_command_spikes=None):
        all_spikes = sensory_spikes
        if motor_command_spikes is not None:
            if sensory_spikes is not None:
                all_spikes = cp.concatenate([sensory_spikes, motor_command_spikes])
            else:
                all_spikes = motor_command_spikes
        self.snn.step(all_spikes)

    def act(self):
        # the brain just observes the consequences of the scripted action.
        pass

    def step(self, stick, ball, motor_command_spikes=None):
        # a full sense-think-act cycle.
        sensory_spikes = self.sense(stick, ball)
        self.think(sensory_spikes, motor_command_spikes)
        self.act()