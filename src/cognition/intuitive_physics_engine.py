from collections import defaultdict

class IntuitivePhysicsEngine:
    def __init__(self, network, pop_info, confidence_threshold=0.5):
        self.network = network
        self.pop_info = pop_info
        self.confidence_threshold = confidence_threshold
        self.causal_graph = defaultdict(dict)
        self.interaction_history = []
        print("[COGNITION] IntuitivePhysicsEngine initialized (Graph-Based)")

    def _get_synapse_strength(self, source_pop, target_pop):
        # extract average weight between two populations from snn
        import cupy as cp
        source = self.pop_info.get(source_pop)
        target = self.pop_info.get(target_pop)
        if source is None or target is None:
            return 0.0
        source_mask = (self.network["source_neurons"] >= source["start"]) & (self.network["source_neurons"] < source["end"])
        target_mask = (self.network["target_neurons"] >= target["start"]) & (self.network["target_neurons"] < target["end"])
        synapse_indices = cp.where(source_mask & target_mask)[0]
        if len(synapse_indices) == 0:
            return 0.0
        avg_weight = cp.mean(self.network["weights"][synapse_indices]).item()
        return avg_weight

    def update_from_observation(self, event_source, event_target, observed_correlation):
        # update belief about causal link based on observation
        current_strength = self.causal_graph[event_source].get(event_target, 0.0)
        # simple bayesian-ish update
        new_strength = 0.8 * current_strength + 0.2 * observed_correlation
        self.causal_graph[event_source][event_target] = new_strength
        self.interaction_history.append({
            "source": event_source,
            "target": event_target,
            "correlation": observed_correlation,
            "updated_strength": new_strength
        })
        if len(self.interaction_history) > 1000:
            self.interaction_history.pop(0)

    def query_causal_link(self, cause_event, effect_event):
        # check if causal link exists with sufficient confidence
        strength = self.causal_graph[cause_event].get(effect_event, 0.0)
        return strength >= self.confidence_threshold, strength

    def get_all_effects(self, cause_event):
        # return all known effects of a cause
        return self.causal_graph[cause_event]

    def get_strongest_cause(self, effect_event):
        # find what most strongly causes an effect
        best_cause = None
        max_strength = 0.0
        for cause, effects in self.causal_graph.items():
            if effect_event in effects and effects[effect_event] > max_strength:
                max_strength = effects[effect_event]
                best_cause = cause
        return best_cause, max_strength

    def extract_beliefs_from_snn(self, object_tracker):
        # rebuild causal graph from current snn weights
        objects = object_tracker.get_all_objects()
        if len(objects) < 2:
            return
        # extract causal chains from snn architecture
        # for embodied_causality: motor → b_moving → a_moving
        causal_chains = [
            ("Franka_Extend_Motor", "B_is_Moving"),
            ("B_is_Moving", "A_is_Moving"),
            ("Franka_Extend_Motor", "A_is_Moving")
        ]
        for source_pop, target_pop in causal_chains:
            strength = self._get_synapse_strength(source_pop, target_pop)
            if strength > 0.01:
                normalized_strength = min(1.0, strength / 5.0)
                self.causal_graph[source_pop][target_pop] = normalized_strength

    def predict_outcome(self, action, target_object_id, object_tracker):
        # predict what will happen if we perform action on target
        obj = object_tracker.get_object(target_object_id)
        if obj is None:
            return "unknown"
        # NOTE: this method has been replaced with `CHG`; but keeping it if we need it down the line for dummy testing.
        action_effects = self.get_all_effects(action)
        if not action_effects:
            return "uncertain"
        # return most likely outcome
        strongest_effect = max(action_effects.items(), key=lambda x: x[1])
        return strongest_effect[0]

    def get_uncertainty(self, action, target_object_id, object_tracker):
        # measure how uncertain we are about outcome
        obj = object_tracker.get_object(target_object_id)
        if obj is None:
            return 1.0
        action_effects = self.get_all_effects(action)
        if not action_effects:
            return 1.0
        max_strength = max(action_effects.values())
        uncertainty = 1.0 - max_strength
        return uncertainty

    def print_beliefs(self):
        print("\n=== IPE BELIEFS ===")
        if not self.causal_graph:
            print("  [No causal beliefs yet]")
            return
        for cause, effects in self.causal_graph.items():
            for effect, strength in effects.items():
                confidence = "HIGH" if strength > 0.7 else "MED" if strength > 0.3 else "LOW"
                print(f"  {cause} → {effect} (strength: {strength:.3f}, confidence: {confidence})")
        print("===================\n")