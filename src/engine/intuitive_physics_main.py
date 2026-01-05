import sys
from pathlib import Path
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from datetime import datetime
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.loader import load_config
from src.network.builder import build_network
from src.engine.simulate import Simulator
try:
    from omni.isaac.kit import SimulationApp
except ImportError:
    print("Error: Isaac Sim libraries not found. Please run this script using the appropriate Isaac Sim launcher (e.g., ./python.sh).")
    sys.exit(1)

def get_synapse_indices(network, pop_info, source_pop_name, target_pop_name):
    source_pop = pop_info[source_pop_name]
    target_pop = pop_info[target_pop_name]
    source_mask = (network["source_neurons"] >= source_pop["start"]) & (network["source_neurons"] < source_pop["end"])
    target_mask = (network["target_neurons"] >= target_pop["start"]) & (network["target_neurons"] < target_pop["end"])
    return cp.where(source_mask & target_mask)[0]

def get_average_weight(network, indices):
    if len(indices) == 0:
        return 0.0
    return cp.mean(network["weights"][indices]).item()

def get_weight_std(network, indices):
    if len(indices) == 0:
        return 0.0
    return cp.std(network["weights"][indices]).item()

def encode_velocity_to_spikes(pop_info, pop_name, velocity_magnitude):
    if velocity_magnitude < 0.1:
        return None
    pop = pop_info[pop_name]
    num_firing = int(pop['count'] * min(0.5, velocity_magnitude / 50.0))
    if num_firing == 0:
        return None
    firing_indices = np.random.choice(
        np.arange(pop['start'], pop['end']), 
        size=num_firing, 
        replace=False
    )
    return cp.asarray(firing_indices, dtype=cp.int32)

def save_learning_plots(weight_history, output_dir):
    plt.figure(figsize=(20, 12))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ax1 = plt.subplot(2, 3, 1)
    trials = list(range(len(weight_history['w1_mean'])))
    ax1.plot(trials, weight_history['w1_mean'], 'b-', linewidth=2, label='Push → A_is_Moving')
    ax1.fill_between(trials, np.array(weight_history['w1_mean']) - np.array(weight_history['w1_std']), np.array(weight_history['w1_mean']) + np.array(weight_history['w1_std']), alpha=0.3, color='blue')
    ax1.plot(trials, weight_history['w2_mean'], 'r-', linewidth=2, label='A_is_Moving → B_is_Moving')
    ax1.fill_between(trials, np.array(weight_history['w2_mean']) - np.array(weight_history['w2_std']), np.array(weight_history['w2_mean']) + np.array(weight_history['w2_std']), alpha=0.3, color='red')
    ax1.set_xlabel('Trial', fontsize=12)
    ax1.set_ylabel('Average Weight', fontsize=12)
    ax1.set_title('Synaptic Weight Evolution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax2 = plt.subplot(2, 3, 2)
    if len(weight_history['w1_mean']) > 1:
        w1_rate = np.diff(weight_history['w1_mean'])
        w2_rate = np.diff(weight_history['w2_mean'])
        ax2.plot(trials[1:], w1_rate, 'b-', linewidth=2, label='Push → A_is_Moving')
        ax2.plot(trials[1:], w2_rate, 'r-', linewidth=2, label='A_is_Moving → B_is_Moving')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Trial', fontsize=12)
        ax2.set_ylabel('Weight Change per Trial', fontsize=12)
        ax2.set_title('Learning Rate', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
    ax3 = plt.subplot(2, 3, 3)
    w1_initial = weight_history['w1_mean'][0]
    w2_initial = weight_history['w2_mean'][0]
    w1_growth = [(w - w1_initial) / w1_initial * 100 for w in weight_history['w1_mean']]
    w2_growth = [(w - w2_initial) / w2_initial * 100 for w in weight_history['w2_mean']]
    ax3.plot(trials, w1_growth, 'b-', linewidth=2, label='Push → A_is_Moving')
    ax3.plot(trials, w2_growth, 'r-', linewidth=2, label='A_is_Moving → B_is_Moving')
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.axhline(y=100, color='g', linestyle='--', alpha=0.5, label='100% growth')
    ax3.set_xlabel('Trial', fontsize=12)
    ax3.set_ylabel('Weight Growth (%)', fontsize=12)
    ax3.set_title('Percentage Weight Increase', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(trials, weight_history['motor_spikes'], 'g-', linewidth=2, label='Motor Spikes')
    ax4.plot(trials, weight_history['sensor_a_spikes'], 'b-', linewidth=2, label='Sensor A Spikes')
    ax4.plot(trials, weight_history['sensor_b_spikes'], 'r-', linewidth=2, label='Sensor B Spikes')
    ax4.set_xlabel('Trial', fontsize=12)
    ax4.set_ylabel('Total Spikes', fontsize=12)
    ax4.set_title('Spike Activity per Trial', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax5 = plt.subplot(2, 3, 5)
    categories = ['Push → A_is_Moving', 'A_is_Moving → B_is_Moving']
    initial_weights = [weight_history['w1_mean'][0], weight_history['w2_mean'][0]]
    final_weights = [weight_history['w1_mean'][-1], weight_history['w2_mean'][-1]]
    x = np.arange(len(categories))
    width = 0.35
    bars1 = ax5.bar(x - width/2, initial_weights, width, label='Initial', color='lightblue')
    bars2 = ax5.bar(x + width/2, final_weights, width, label='Final', color='darkblue')
    ax5.set_ylabel('Average Weight', fontsize=12)
    ax5.set_title('Initial vs Final Weights', fontsize=14, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(categories, fontsize=9)
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3, axis='y')
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    w1_fold_change = weight_history['w1_mean'][-1] / weight_history['w1_mean'][0]
    w2_fold_change = weight_history['w2_mean'][-1] / weight_history['w2_mean'][0]
    total_trials = len(weight_history['w1_mean'])
    summary_text = f"""
    LEARNING SUMMARY
    ═══════════════════════════════
    
    Total Trials: {total_trials}
    
    Push → A_is_Moving:
      Initial: {weight_history['w1_mean'][0]:.4f}
      Final: {weight_history['w1_mean'][-1]:.4f}
      Fold Change: {w1_fold_change:.2f}x
      % Increase: {(w1_fold_change-1)*100:.1f}%
    
    A_is_Moving → B_is_Moving:
      Initial: {weight_history['w2_mean'][0]:.4f}
      Final: {weight_history['w2_mean'][-1]:.4f}
      Fold Change: {w2_fold_change:.2f}x
      % Increase: {(w2_fold_change-1)*100:.1f}%
    
    Learning Status:
      {'✓ SUCCESS' if w1_fold_change > 2 and w2_fold_change > 2 else '✗ FAILURE'}

    Total Spikes Generated:
      Motor: {sum(weight_history['motor_spikes']):,}
      Sensor A: {sum(weight_history['sensor_a_spikes']):,}
      Sensor B: {sum(weight_history['sensor_b_spikes']):,}
    """
    ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace', verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plot_path = output_dir / f"learning_analysis_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n[VISUALIZATION] Saved comprehensive learning plot to: {plot_path}")
    csv_path = output_dir / f"learning_data_{timestamp}.csv"
    with open(csv_path, 'w') as f:
        f.write("Trial,W1_Mean,W1_Std,W2_Mean,W2_Std,Motor_Spikes,Sensor_A_Spikes,Sensor_B_Spikes\n")
        for i in range(len(weight_history['w1_mean'])):
            f.write(f"{i},{weight_history['w1_mean'][i]:.6f},{weight_history['w1_std'][i]:.6f},"
                f"{weight_history['w2_mean'][i]:.6f},{weight_history['w2_std'][i]:.6f},"
                f"{weight_history['motor_spikes'][i]},{weight_history['sensor_a_spikes'][i]},"
                f"{weight_history['sensor_b_spikes'][i]}\n"
            )
    print(f"[VISUALIZATION] Saved raw data to: {csv_path}")
    plt.close()

def run_simulation(simulation_app):
    print("[DEBUG] run_simulation coroutine started.")
    from omni.isaac.core import World
    from omni.isaac.core.objects import DynamicSphere
    from pxr import Gf
    base_path = Path(__file__).resolve().parent.parent.parent
    network_config = load_config(base_path / "config" / "network" / "intuitive_physics_test.json")
    sim_config = load_config(base_path / "config" / "simulation" / "default_run.json")
    print("Building Intuitive Physics Engine (IPE) network...")
    network, pop_info = build_network(network_config)
    print(f"Network built with {network['membrane_potential'].shape[0]} neurons and {network['source_neurons'].shape[0]} synapses.")
    # SNN sim.
    print("[DEBUG] Creating SNN Simulator object...")
    snn_simulator = Simulator(network, pop_info, sim_config)
    print("[DEBUG] SNN Simulator object created.")
    push_to_moving_a_synapses = get_synapse_indices(network, pop_info, "Push_A_Motor", "A_is_Moving")
    moving_a_to_moving_b_synapses = get_synapse_indices(network, pop_info, "A_is_Moving", "B_is_Moving")
    weight_history = {
        'w1_mean': [],
        'w1_std': [],
        'w2_mean': [],
        'w2_std': [],
        'motor_spikes': [],
        'sensor_a_spikes': [],
        'sensor_b_spikes': []
    }
    w1_initial = get_average_weight(network, push_to_moving_a_synapses)
    w2_initial = get_average_weight(network, moving_a_to_moving_b_synapses)
    print(f"\nInitial Avg. Weight (Push -> A_is_Moving): {w1_initial:.4f}")
    print(f"Initial Avg. Weight (A_is_Moving -> B_is_Moving): {w2_initial:.4f}\n")
    # world setup.
    print("[DEBUG] Setting up Isaac Sim world...")
    world = World()
    world.scene.add_default_ground_plane()
    # create the billiard balls
    ball_a = world.scene.add(
        DynamicSphere(
            prim_path="/World/ball_a",
            name="ball_a",
            position=np.array([0, -0.5, 0.2]),
            radius=0.1,
            color=np.array([1.0, 0.0, 0.0])
        )
    )
    ball_b = world.scene.add(
        DynamicSphere(
            prim_path="/World/ball_b",
            name="ball_b",
            position=np.array([0, 0.5, 0.2]),
            radius=0.1,
            color=np.array([0.0, 0.0, 1.0])
        )
    )
    print("[DEBUG] Resetting world...")
    world.reset()
    print("[DEBUG] world.reset() has completed.")
    num_trials = 50
    # each trial is 200 ms.
    trial_duration_ms = 200
    trial_steps = int(trial_duration_ms / sim_config['dt'])
    sample_interval = 1
    print(f"=== STARTING TRAINING: {num_trials} trials ===")
    for i in range(num_trials):
        print(f"Running Trial {i+1}/{num_trials}")
        ball_a.set_world_pose(position=np.array([0, -0.5, 0.2]))
        ball_b.set_world_pose(position=np.array([0, 0.5, 0.2]))
        ball_a.set_linear_velocity(np.zeros(3))
        ball_b.set_linear_velocity(np.zeros(3))
        for _ in range(10): 
            world.step(render=True)
        # track spikes for this trial
        trial_motor_spikes = 0
        trial_sensor_a_spikes = 0
        trial_sensor_b_spikes = 0
        for step in range(trial_steps):
            print(f"\r    -> Trial {i+1}, Step {step+1}/{trial_steps}", end="")
            current_time_ms = step * sim_config['dt']
            motor_spikes = None
            # action.
            if 0 <= current_time_ms < 10:
                impulse_velocity = np.array([0, 2.5, 0])
                ball_a.set_linear_velocity(impulse_velocity)
                pop = pop_info['Push_A_Motor']
                motor_spikes = cp.arange(pop['start'], pop['end'], dtype=cp.int32)
                trial_motor_spikes += len(motor_spikes)
            # sense.
            vel_a = ball_a.get_linear_velocity()
            vel_b = ball_b.get_linear_velocity()
            spikes_a = encode_velocity_to_spikes(pop_info, "Object_A_Detector", np.linalg.norm(vel_a))
            spikes_b = encode_velocity_to_spikes(pop_info, "Object_B_Detector", np.linalg.norm(vel_b))
            
            if spikes_a is not None:
                trial_sensor_a_spikes += len(spikes_a)
            if spikes_b is not None:
                trial_sensor_b_spikes += len(spikes_b)
            spike_lists = [s for s in [motor_spikes, spikes_a, spikes_b] if s is not None]
            all_sensory_spikes = cp.concatenate(spike_lists) if spike_lists else None
            # think.
            snn_simulator.step(all_sensory_spikes)
            # act.
            world.step(render=True)
        print()
            
        # sample weights after each trial
        if (i + 1) % sample_interval == 0:
            w1_mean = get_average_weight(network, push_to_moving_a_synapses)
            w1_std = get_weight_std(network, push_to_moving_a_synapses)
            w2_mean = get_average_weight(network, moving_a_to_moving_b_synapses)
            w2_std = get_weight_std(network, moving_a_to_moving_b_synapses)
            weight_history['w1_mean'].append(w1_mean)
            weight_history['w1_std'].append(w1_std)
            weight_history['w2_mean'].append(w2_mean)
            weight_history['w2_std'].append(w2_std)
            weight_history['motor_spikes'].append(trial_motor_spikes)
            weight_history['sensor_a_spikes'].append(trial_sensor_a_spikes)
            weight_history['sensor_b_spikes'].append(trial_sensor_b_spikes)
            print(f"    [DEBUG] W1: {w1_mean:.4f} (±{w1_std:.4f}), W2: {w2_mean:.4f} (±{w2_std:.4f})")
            print(f"    [DEBUG] Spikes - Motor: {trial_motor_spikes}, Sensor A: {trial_sensor_a_spikes}, Sensor B: {trial_sensor_b_spikes}")
    print("=== TRAINING COMPLETE ===")
    world.pause()
    w1_final = get_average_weight(network, push_to_moving_a_synapses)
    w2_final = get_average_weight(network, moving_a_to_moving_b_synapses)
    print("\n=== LEARNING VERIFICATION ===")
    print(f"Avg. Weight (Push -> A_is_Moving): {w1_initial:.4f} -> {w1_final:.4f}")
    print(f"Avg. Weight (A_is_Moving -> B_is_Moving): {w2_initial:.4f} -> {w2_final:.4f}\n")
    if w1_final > w1_initial * 2 and w2_final > w2_initial * 2:
        print("SUCCESS: Causal links were learned and strengthened via STDP.")
    else:
        print("FAILURE: Synaptic weights did not increase significantly. Learning did not occur.")
    output_dir = base_path / "results" / "visualizations"
    save_learning_plots(weight_history, output_dir)

if __name__ == "__main__":
    simulation_app = SimulationApp({"headless": False})
    print("[DEBUG] SimulationApp object created. Waiting for Isaac Sim to initialize...")
    try:
        run_simulation(simulation_app)
    except Exception as e:
        print(f"An error occurred during simulation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[DEBUG] Simulation finished or error occurred. Closing SimulationApp.")
        simulation_app.close()
