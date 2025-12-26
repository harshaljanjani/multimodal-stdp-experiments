import cupy as cp
import topology

def run_test(test_func):
    try:
        test_func()
        print(f"[PASS] {test_func.__name__}")
    except Exception as e:
        print(f"[FAIL] {test_func.__name__}: {e}")

def test_generate_grid_positions():
    positions = topology.generate_grid_positions(100)
    assert positions.shape == (100, 2), "Shape should be (100, 2)"
    positions = topology.generate_grid_positions(99)
    assert positions.shape == (99, 2), "Shape should be (99, 2) for non-perfect squares"
    assert isinstance(positions, cp.ndarray), "Should return a CuPy array"

def test_fixed_probability_connections():
    s, t = topology.create_fixed_probability_connections(100, 200, 0.1)
    expected_synapses = 100 * 200 * 0.1
    assert abs(len(s) - expected_synapses) < 500, "Synapse count should be close to expected"
    assert s.max() < 100
    assert t.max() < 200

def test_autapse_prevention():
    s, t = topology.create_fixed_probability_connections(
        1000, 1000, 0.5, allow_autapses=False
    )
    autapses = cp.sum(s == t)
    assert autapses == 0, "No autapses should be present when disallowed"
    s_allowed, t_allowed = topology.create_fixed_probability_connections(
        1000, 1000, 0.5, allow_autapses=True
    )
    assert cp.sum(s_allowed == t_allowed) > 0, "Autapses should be present when allowed"

def test_gaussian_distance_connections():
    pos = topology.generate_grid_positions(100)
    s, t = topology.create_gaussian_distance_connections(pos, pos, p_max=1.0, sigma=0.01)
    distances = cp.sqrt(cp.sum((pos[s] - pos[t])**2, axis=1))
    assert cp.all(distances < 0.1), "With tiny sigma, all connections must be very short"
    s_rand, _ = topology.create_gaussian_distance_connections(pos, pos, p_max=0.1, sigma=100.0)
    expected_synapses = 100 * 100 * 0.1
    assert abs(len(s_rand) - expected_synapses) < 500, "With large sigma, synapse count should approach p_max"

def main():
    print("Running topology sanity checks...")
    run_test(test_generate_grid_positions)
    run_test(test_fixed_probability_connections)
    run_test(test_autapse_prevention)
    run_test(test_gaussian_distance_connections)
    print("...done.")

main()
