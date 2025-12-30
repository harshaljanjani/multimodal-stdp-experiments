import cupy as cp
import numpy as np
from typing import Any, Dict, List, Tuple
from src.network import topology

def _parse_params(param_config: Any, num_items: int) -> cp.ndarray:
    if isinstance(param_config, (int, float)):
        return cp.full(num_items, param_config, dtype=cp.float32)
    dist = param_config.get("distribution")
    if dist == "normal":
        mean = param_config["mean"]
        std = param_config["std"]
        if std < 0:
            raise ValueError("Standard deviation cannot be negative.")
        return cp.random.normal(loc=mean, scale=std, size=num_items).astype(cp.float32)
    elif dist == "uniform":
        min_val = param_config["min"]
        max_val = param_config["max"]
        return cp.random.uniform(min_val, max_val, num_items).astype(cp.float32)
    raise ValueError(f"Unknown parameter format: {param_config}")

def build_network(config: Dict[str, Any]) -> Tuple[Dict[str, cp.ndarray], Dict[str, Any]]:
    pop_names = [p["name"] for p in config["neuron_populations"]]
    if len(pop_names) != len(set(pop_names)):
        raise ValueError("Duplicate population names found in configuration.")
    total_neurons = sum(p["count"] for p in config["neuron_populations"])
    neuron_states = {
        "membrane_potential": cp.zeros(total_neurons, dtype=cp.float32),
        "refractory_time": cp.zeros(total_neurons, dtype=cp.float32),
        "trace_pre": cp.zeros(total_neurons, dtype=cp.float32),
        "trace_post": cp.zeros(total_neurons, dtype=cp.float32)
    }
    neuron_params = {}
    pop_info = {}
    pop_positions = {}
    current_offset = 0
    for pop in config["neuron_populations"]:
        name = pop["name"]
        count = pop["count"]
        start = current_offset
        end = current_offset + count
        pop_type = pop.get("type", "lif")
        pop_info[name] = {"start": start, "end": end, "count": count, "type": pop_type}
        if "positions" in pop:
            pos_config = pop["positions"]
            if pos_config["type"] == "grid":
                pop_positions[name] = topology.generate_grid_positions(count)
            elif pos_config["type"] == "random":
                pop_positions[name] = topology.generate_random_positions(count)
            else:
                raise ValueError(f"Unknown position generation type: {pos_config['type']}")
        if count > 0:
            if "params" in pop:
                for param, value in pop["params"].items():
                    if param not in neuron_params:
                        neuron_params[param] = cp.zeros(total_neurons, dtype=cp.float32)
                    neuron_params[param][start:end] = _parse_params(value, count)
        current_offset += count
    if "v_leak" in neuron_params:
        neuron_states["membrane_potential"] = cp.copy(neuron_params["v_leak"])
    all_source_ids: List[np.ndarray] = []
    all_target_ids: List[np.ndarray] = []
    all_weights: List[cp.ndarray] = []
    all_delays: List[cp.ndarray] = []
    all_learning_rates: List[cp.ndarray] = []
    all_max_weights: List[cp.ndarray] = []
    all_prune_thresholds: List[cp.ndarray] = []
    for conn in config["synaptic_connections"]:
        source_info = pop_info[conn["source"]]
        target_info = pop_info[conn["target"]]
        if source_info["count"] == 0 or target_info["count"] == 0:
            continue
        conn_topo = conn["topology"]
        allow_autapses = conn_topo.get("allow_autapses", False)
        is_recurrent = conn["source"] == conn["target"]
        if conn_topo["type"] == "fixed_probability":
            source_indices, target_indices = topology.create_fixed_probability_connections(
                source_info["count"],
                target_info["count"],
                conn_topo["probability"],
                allow_autapses,
                is_recurrent
            )
        elif conn_topo["type"] == "gaussian_distance":
            source_pos = pop_positions[conn["source"]]
            target_pos = pop_positions[conn["target"]]
            source_indices, target_indices = topology.create_gaussian_distance_connections(
                source_pos, target_pos, conn_topo["p_max"], conn_topo["sigma"], allow_autapses)
        else:
            raise ValueError(f"Unknown topology type: {conn_topo['type']}")
        num_new_synapses = len(source_indices)
        if num_new_synapses == 0:
            continue
        all_source_ids.append(source_indices + source_info["start"])
        all_target_ids.append(target_indices + target_info["start"])
        synapse_config = conn["synapse"]
        all_weights.append(_parse_params(synapse_config["weight"], num_new_synapses))
        all_delays.append(cp.maximum(1.0, cp.rint(_parse_params(synapse_config["delay"], num_new_synapses))))
        prune_thresh = _parse_params(synapse_config.get("prune_threshold", 0.0), num_new_synapses)
        all_prune_thresholds.append(prune_thresh)
        plasticity_config = synapse_config.get("plasticity")
        if plasticity_config:
            all_learning_rates.append(_parse_params(plasticity_config["learning_rate"], num_new_synapses))
            all_max_weights.append(_parse_params(plasticity_config["max_weight"], num_new_synapses))
        else:
            all_learning_rates.append(cp.zeros(num_new_synapses, dtype=cp.float32))
            all_max_weights.append(cp.full(num_new_synapses, cp.inf, dtype=cp.float32))
    synapse_data = {
        "source_neurons": cp.asarray(np.concatenate(all_source_ids), dtype=cp.int32) if all_source_ids else cp.array([], dtype=cp.int32),
        "target_neurons": cp.asarray(np.concatenate(all_target_ids), dtype=cp.int32) if all_target_ids else cp.array([], dtype=cp.int32),
        "weights": cp.concatenate(all_weights) if all_weights else cp.array([], dtype=cp.float32),
        "delays": cp.concatenate(all_delays).astype(cp.int32) if all_delays else cp.array([], dtype=cp.int32),
        "learning_rate": cp.concatenate(all_learning_rates) if all_learning_rates else cp.array([], dtype=cp.float32),
        "max_weight": cp.concatenate(all_max_weights) if all_max_weights else cp.array([], dtype=cp.float32),
        "prune_threshold": cp.concatenate(all_prune_thresholds) if all_prune_thresholds else cp.array([], dtype=cp.float32)
    }
    if pop_positions:
        all_positions = cp.vstack([pop_positions[p["name"]] for p in config["neuron_populations"] if p["name"] in pop_positions])
        synapse_data["neuron_positions"] = all_positions
    network = {**neuron_states, **neuron_params, **synapse_data}
    if "tau_trace_pre" not in network:
        network["tau_trace_pre"] = cp.full(total_neurons, 20.0, dtype=cp.float32)
    if "tau_trace_post" not in network:
        network["tau_trace_post"] = cp.full(total_neurons, 20.0, dtype=cp.float32)
    if "i_background" not in network:
         network["i_background"] = cp.zeros(total_neurons, dtype=cp.float32)
    return network, pop_info
