import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

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

def save_curiosity_plots(history, output_dir, experiment_name="curiosity_learning"):
    plt.figure(figsize=(15, 8))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # plot 1: synaptic weight evolution
    ax1 = plt.subplot(1, 2, 1)
    steps = list(range(len(next(iter(history['weights'].values())))))
    colors = plt.cm.jet(np.linspace(0, 1, len(history['weights'])))
    for i, (name, weights) in enumerate(history['weights'].items()):
        ax1.plot(steps, weights, label=name, color=colors[i], linewidth=2)
    ax1.set_xlabel('Sample Point (x100 steps)', fontsize=12)
    ax1.set_ylabel('Average Synaptic Weight', fontsize=12)
    ax1.set_title('Hebbian Learning: Weight Evolution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.4)
    # plot 2: spike activity
    ax2 = plt.subplot(1, 2, 2)
    steps = list(range(len(next(iter(history['spikes'].values())))))
    colors = plt.cm.jet(np.linspace(0, 1, len(history['spikes'])))
    for i, (name, spikes) in enumerate(history['spikes'].items()):
         ax2.plot(steps, spikes, label=name, color=colors[i], linewidth=2)
    ax2.set_xlabel('Sample Point (x100 steps)', fontsize=12)
    ax2.set_ylabel('Total Spikes per Sample Window', fontsize=12)
    ax2.set_title('Population Spike Activity', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.4)
    plt.tight_layout()
    plot_path = output_dir / f"{experiment_name}_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n[VISUALIZATION] Saved learning plot to: {plot_path}")
    plt.close()

def save_learning_plots(weight_history, output_dir, experiment_name="learning", connection1_label="Connection 1", connection2_label="Connection 2"):
    plt.figure(figsize=(20, 12))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # plot 1: weight evolution
    ax1 = plt.subplot(2, 3, 1)
    trials = list(range(len(weight_history['w1_mean'])))
    ax1.plot(trials, weight_history['w1_mean'], 'b-', linewidth=2, label=connection1_label)
    ax1.fill_between(trials, np.array(weight_history['w1_mean']) - np.array(weight_history['w1_std']), np.array(weight_history['w1_mean']) + np.array(weight_history['w1_std']), alpha=0.3, color='blue')
    ax1.plot(trials, weight_history['w2_mean'], 'r-', linewidth=2, label=connection2_label)
    ax1.fill_between(trials, np.array(weight_history['w2_mean']) - np.array(weight_history['w2_std']), np.array(weight_history['w2_mean']) + np.array(weight_history['w2_std']), alpha=0.3, color='red')
    ax1.set_xlabel('Trial', fontsize=12)
    ax1.set_ylabel('Average Weight', fontsize=12)
    ax1.set_title('Synaptic Weight Evolution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    # plot 2: learning rate
    ax2 = plt.subplot(2, 3, 2)
    if len(weight_history['w1_mean']) > 1:
        w1_rate = np.diff(weight_history['w1_mean'])
        w2_rate = np.diff(weight_history['w2_mean'])
        ax2.plot(trials[1:], w1_rate, 'b-', linewidth=2, label=connection1_label)
        ax2.plot(trials[1:], w2_rate, 'r-', linewidth=2, label=connection2_label)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Trial', fontsize=12)
        ax2.set_ylabel('Weight Change per Trial', fontsize=12)
        ax2.set_title('Learning Rate', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
    # plot 3: percentage growth
    ax3 = plt.subplot(2, 3, 3)
    w1_initial = weight_history['w1_mean'][0]
    w2_initial = weight_history['w2_mean'][0]
    w1_growth = [(w - w1_initial) / w1_initial * 100 if w1_initial > 0 else 0 for w in weight_history['w1_mean']]
    w2_growth = [(w - w2_initial) / w2_initial * 100 if w2_initial > 0 else 0 for w in weight_history['w2_mean']]
    ax3.plot(trials, w1_growth, 'b-', linewidth=2, label=connection1_label)
    ax3.plot(trials, w2_growth, 'r-', linewidth=2, label=connection2_label)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.axhline(y=100, color='g', linestyle='--', alpha=0.5, label='100% growth')
    ax3.set_xlabel('Trial', fontsize=12)
    ax3.set_ylabel('Weight Growth (%)', fontsize=12)
    ax3.set_title('Percentage Weight Increase', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    # plot 4: spike activity
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(trials, weight_history['motor_spikes'], 'g-', linewidth=2, label='Motor Spikes')
    ax4.plot(trials, weight_history['sensor_a_spikes'], 'b-', linewidth=2, label='Sensor A Spikes')
    ax4.plot(trials, weight_history['sensor_b_spikes'], 'r-', linewidth=2, label='Sensor B Spikes')
    ax4.set_xlabel('Trial', fontsize=12)
    ax4.set_ylabel('Total Spikes', fontsize=12)
    ax4.set_title('Spike Activity per Trial', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    # plot 5: initial vs final
    ax5 = plt.subplot(2, 3, 5)
    categories = [connection1_label, connection2_label]
    initial_weights = [weight_history['w1_mean'][0], weight_history['w2_mean'][0]]
    final_weights = [weight_history['w1_mean'][-1], weight_history['w2_mean'][-1]]
    x = np.arange(len(categories))
    width = 0.35
    bars1 = ax5.bar(x - width/2, initial_weights, width, label='Initial', color='lightblue')
    bars2 = ax5.bar(x + width/2, final_weights, width, label='Final', color='darkblue')
    ax5.set_ylabel('Average Weight', fontsize=12)
    ax5.set_title('Initial vs Final Weights', fontsize=14, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(categories, fontsize=9, rotation=15, ha='right')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3, axis='y')
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', 
                    ha='center', va='bottom', fontsize=9)
    # plot 6: summary stats
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    w1_fold_change = weight_history['w1_mean'][-1] / weight_history['w1_mean'][0] if weight_history['w1_mean'][0] > 0 else 0
    w2_fold_change = weight_history['w2_mean'][-1] / weight_history['w2_mean'][0] if weight_history['w2_mean'][0] > 0 else 0
    total_trials = len(weight_history['w1_mean'])
    
    summary_text = f"""
    {experiment_name.upper()} LEARNING SUMMARY
    ═══════════════════════════════════════
    
    Total Trials: {total_trials}
    
    {connection1_label}:
      Initial: {weight_history['w1_mean'][0]:.4f}
      Final: {weight_history['w1_mean'][-1]:.4f}
      Fold Change: {w1_fold_change:.2f}x
      % Increase: {(w1_fold_change-1)*100:.1f}%
    
    {connection2_label}:
      Initial: {weight_history['w2_mean'][0]:.4f}
      Final: {weight_history['w2_mean'][-1]:.4f}
      Fold Change: {w2_fold_change:.2f}x
      % Increase: {(w2_fold_change-1)*100:.1f}%
    
    Learning Status:
      {'✓ SUCCESS' if w1_fold_change > 1.5 and w2_fold_change > 1.5 else '✗ FAILURE'}
    
    Total Spikes Generated:
      Motor: {sum(weight_history['motor_spikes']):,}
      Sensor A: {sum(weight_history['sensor_a_spikes']):,}
      Sensor B: {sum(weight_history['sensor_b_spikes']):,}
    """
    ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace', verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plot_path = output_dir / f"{experiment_name}_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n[VISUALIZATION] Saved comprehensive learning plot to: {plot_path}")
    csv_path = output_dir / f"{experiment_name}_data_{timestamp}.csv"
    with open(csv_path, 'w') as f:
        f.write("Trial,W1_Mean,W1_Std,W2_Mean,W2_Std,Motor_Spikes,Sensor_A_Spikes,Sensor_B_Spikes\n")
        for i in range(len(weight_history['w1_mean'])):
            f.write(
                f"{i},{weight_history['w1_mean'][i]:.6f},{weight_history['w1_std'][i]:.6f},"
                f"{weight_history['w2_mean'][i]:.6f},{weight_history['w2_std'][i]:.6f},"
                f"{weight_history['motor_spikes'][i]},{weight_history['sensor_a_spikes'][i]},"
                f"{weight_history['sensor_b_spikes'][i]}\n"
            )
    print(f"[VISUALIZATION] Saved raw data to: {csv_path}")
    plt.close()

def print_learning_results(w1_initial, w1_final, w2_initial, w2_final, connection1_label, connection2_label, threshold=1.5):
    print("\n=== LEARNING VERIFICATION ===")
    print(f"{connection1_label}: {w1_initial:.4f} → {w1_final:.4f}")
    print(f"{connection2_label}: {w2_initial:.4f} → {w2_final:.4f}\n") 
    w1_fold_change = w1_final / w1_initial if w1_initial > 0 else 0
    w2_fold_change = w2_final / w2_initial if w2_initial > 0 else 0
    if w1_fold_change > threshold and w2_fold_change > threshold:
        print("✓ SUCCESS: Causal links were learned and strengthened via STDP.")
        print(f"  {connection1_label} increased by {(w1_fold_change-1)*100:.1f}%")
        print(f"  {connection2_label} increased by {(w2_fold_change-1)*100:.1f}%")
    else:
        print("✗ FAILURE: Synaptic weights did not increase significantly. Learning did not occur.")
        print(f"  {connection1_label} fold change: {w1_fold_change:.2f}x (need >{threshold}x)")
        print(f"  {connection2_label} fold change: {w2_fold_change:.2f}x (need >{threshold}x)")