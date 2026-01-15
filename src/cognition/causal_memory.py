import collections

class CausalMemory:
    def __init__(self):
        # simple graph structure: {cause: {effect: strength}}
        self.causal_graph = collections.defaultdict(dict)
        print("[COGNITION] CausalMemory initialized (Graph-based v0)")

    def add_or_strengthen_link(self, cause_event, effect_event, strength_update):
        current_strength = self.causal_graph[cause_event].get(effect_event, 0.0)
        # naive update rule, just adds the strength. can be improved later.
        self.causal_graph[cause_event][effect_event] = current_strength + strength_update

    def get_strongest_cause(self, effect_event):
        best_cause = None
        max_strength = -1.0
        for cause, effects in self.causal_graph.items():
            if effect_event in effects and effects[effect_event] > max_strength:
                max_strength = effects[effect_event]
                best_cause = cause
        return best_cause, max_strength

    def print_memory(self):
        print("\n=== Causal Memory Contents ===")
        if not self.causal_graph:
            print("  [Empty]")
            return
        for cause, effects in self.causal_graph.items():
            for effect, strength in effects.items():
                print(f"  {cause} → {effect} (Strength: {strength:.4f})")
        print("============================\n")