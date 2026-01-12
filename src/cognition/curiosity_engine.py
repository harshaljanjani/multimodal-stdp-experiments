import numpy as np

class CuriosityEngine:
    def __init__(self, action_space):
        self.action_space = action_space
        print("[COGNITION] v0 Curiosity Engine initialized (Random Action Policy)")

    def step(self, sensory_input, motor_rates):
        # TODO: this is a placeholder for a real prediction-error model.
        action_index = np.random.randint(0, len(self.action_space))
        return self.action_space[action_index]