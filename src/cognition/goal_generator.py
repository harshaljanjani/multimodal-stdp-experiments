import numpy as np
from collections import deque

class GoalGenerator:
    def __init__(self, action_space):
        self.action_space = action_space
        self.active_goal = None
        self.goal_history = deque(maxlen=50)
        self.goal_types = ["explore_object", "test_causality", "repeat_surprising"]
        print("[COGNITION] GoalGenerator initialized (Information-Gain Driven)")

    def generate_goal(self, object_tracker, ipe, curiosity_engine):
        # generate next goal based on current knowledge state
        objects = object_tracker.get_all_objects()
        if len(objects) == 0:
            return self._create_goal("explore_space", None, None)
        # strategy 1: find object with highest uncertainty
        max_uncertainty = 0.0
        most_uncertain_obj = None
        for obj_id, _ in objects.items():
            for action in self.action_space:
                uncertainty = ipe.get_uncertainty(action, obj_id, object_tracker)
                if uncertainty > max_uncertainty:
                    max_uncertainty = uncertainty
                    most_uncertain_obj = obj_id
        # strategy 2: check if curiosity engine has high prediction error
        recent_pred_errors = curiosity_engine.prediction_errors[-10:] if curiosity_engine.prediction_errors else []
        avg_pred_error = np.mean(recent_pred_errors) if recent_pred_errors else 0.0
        # prioritize uncertainty-based exploration
        if max_uncertainty > 0.7 and most_uncertain_obj is not None:
            action = np.random.choice(self.action_space)
            return self._create_goal("test_causality", most_uncertain_obj, action)
        # fallback: explore space
        return self._create_goal("explore_space", None, None)

    def _create_goal(self, goal_type, target_object_id, action):
        goal = {
            "type": goal_type,
            "target": target_object_id,
            "action": action,
            "created_at": len(self.goal_history)
        }
        self.goal_history.append(goal)
        self.active_goal = goal
        return goal

    def get_active_goal(self):
        return self.active_goal

    def mark_goal_complete(self, outcome):
        if self.active_goal is not None:
            self.active_goal["outcome"] = outcome
            self.active_goal["completed"] = True
        self.active_goal = None

    def print_goal(self, goal):
        if goal is None:
            print("[GOAL] No active goal")
            return
        print(f"[GOAL] type={goal['type']} | target={goal.get('target')} | action={goal.get('action')}")
