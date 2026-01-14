import numpy as np
import cupy as cp
from src.cognition.recovery_detector import RecoveryDetector

class CuriosityEngine:
    def __init__(self, action_space, sensory_dim=128):
        self.action_space = action_space
        self.sensory_dim = sensory_dim
        self.prediction_window = 50
        self.sensory_history = []
        self.prediction_errors = []
        self.epsilon = 0.25
        self.action_history = []
        self.action_window = 10
        self.recovery_detector = RecoveryDetector()
        self.recovery_mode = False
        self.recovery_steps = 0
        self.recovery_sequence = ["turn_right"] * 8 + ["forward"] * 3
        self.recovery_cooldown = 0
        print("[COGNITION] v0 Curiosity Engine initialized (Prediction-Error Driven)")

    def _compute_sensory_summary(self, img_gpu):
        if img_gpu is None or img_gpu.shape[0] == 0:
            return cp.zeros(self.sensory_dim, dtype=cp.float32)
        # simple summary: downsample and flatten
        h, w, c = img_gpu.shape
        target_h, target_w = 8, 8
        step_h, step_w = max(1, h // target_h), max(1, w // target_w)
        downsampled = img_gpu[::step_h, ::step_w, :]
        flattened = downsampled.flatten()
        if flattened.shape[0] > self.sensory_dim:
            summary = flattened[:self.sensory_dim]
        else:
            summary = cp.zeros(self.sensory_dim, dtype=cp.float32)
            summary[:flattened.shape[0]] = flattened
        return summary

    def _predict_next_state(self):
        if len(self.sensory_history) < 2:
            return self.sensory_history[-1] if self.sensory_history else cp.zeros(self.sensory_dim, dtype=cp.float32)
        recent = self.sensory_history[-min(5, len(self.sensory_history)):]
        velocity = cp.mean(cp.array([recent[i+1] - recent[i] for i in range(len(recent)-1)]), axis=0)
        prediction = self.sensory_history[-1] + velocity
        return prediction

    def _compute_prediction_error(self, actual):
        if len(self.sensory_history) < 2:
            return 0.0
        predicted = self._predict_next_state()
        error = float(cp.linalg.norm(actual - predicted))
        return error

    def step(self, sensory_input, robot_position, motion_intensity=0.0):
        sensory_summary = self._compute_sensory_summary(sensory_input)
        pred_error = self._compute_prediction_error(sensory_summary)
        self.prediction_errors.append(pred_error)
        if len(self.prediction_errors) > self.prediction_window:
            self.prediction_errors.pop(0)
        self.sensory_history.append(sensory_summary)
        if len(self.sensory_history) > self.prediction_window:
            self.sensory_history.pop(0)
        current_action = self.action_history[-1] if self.action_history else "stop"
        self.recovery_detector.update(robot_position, current_action)
        if self.recovery_cooldown > 0:
            self.recovery_cooldown -= 1
        if self.recovery_mode:
            action = self.recovery_sequence[self.recovery_steps]
            self.recovery_steps += 1
            if self.recovery_steps >= len(self.recovery_sequence):
                self.recovery_mode = False
                self.recovery_steps = 0
                self.recovery_cooldown = 50
                print("[CURIOSITY] Recovery complete, cooldown active")
        elif self.recovery_detector.is_stuck() and self.recovery_cooldown == 0:
            self.recovery_mode = True
            self.recovery_steps = 0
            action = self.recovery_sequence[0]
            print("[CURIOSITY] WALL DETECTED! Executing 180-degree turn")
        else:
            # exploit: choose action that historically led to high prediction error
            if np.random.rand() < self.epsilon:
                action_index = np.random.randint(0, len(self.action_space))
            else:
                if len(self.action_history) < self.action_window:
                    action_index = np.random.randint(0, len(self.action_space))
                else:
                    action_counts = {a: self.action_history[-self.action_window:].count(a) for a in range(len(self.action_space))}
                    action_index = min(action_counts, key=action_counts.get)
            action = self.action_space[action_index]
        self.action_history.append(action)
        if len(self.action_history) > self.prediction_window:
            self.action_history.pop(0)
        return action

    def get_most_surprising_object(self, object_tracker):
        objects = object_tracker.get_all_objects()
        if not objects:
            return None
        max_surprise = 0.0
        most_surprising = None
        for obj_id, obj_data in objects.items():
            vel_magnitude = np.linalg.norm(obj_data["velocity"])
            if vel_magnitude > max_surprise:
                max_surprise = vel_magnitude
                most_surprising = obj_id
        return most_surprising

    def propose_exploration_goal(self, object_tracker, ipe):
        objects = object_tracker.get_all_objects()
        if not objects:
            return "explore"
        max_uncertainty = 0.0
        target_obj = None
        for obj_id in objects.keys():
            uncertainty = ipe.get_uncertainty("push", obj_id, object_tracker)
            if uncertainty > max_uncertainty:
                max_uncertainty = uncertainty
                target_obj = obj_id
        if target_obj is not None and max_uncertainty > 0.5:
            return f"approach_object_{target_obj}"
        return "explore"
