import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from core.node import CorticalColumn

def run_experiment():
    dt = 0.01
    duration = 15.0
    time_steps = int(duration / dt)
    t = np.linspace(0, duration, time_steps)
    
    # Input: Sine Wave
    input_signal = np.sin(2.0 * t)
    
    # Occlusion (Blind Spot) from t=8 to t=12
    input_precision = np.ones_like(t)
    cut_start = int(8.0 / dt)
    cut_end = int(12.0 / dt)
    input_precision[cut_start:cut_end] = 0.0
    
    # Hierarchy
    # L1: Fast sensory tracker
    layer1 = CorticalColumn(order=2, dt=dt, learning_rate=10.0, w_learning_rate=0.05)
    
    # L2: Slower, deeper context
    # L2 tracks L1's state. L2 should have lower learning rate? 
    # Or L2 tracks the *average*?
    # For this simple test, L2 is just another column stacking on top.
    layer2 = CorticalColumn(order=2, dt=dt, learning_rate=5.0, w_learning_rate=0.01)
    
    l1_history = []
    l2_history = []
    
    print("Starting Phase 3: Hierarchy...")
    for i in range(time_steps):
        val = input_signal[i]
        prec = input_precision[i]
        
        # --- Step 1: L2 predicts L1 (Top-Down) ---
        # L2's state IS the prediction for L1.
        # But L2 tracks L1's motion. 
        # So L2.mu predicts L1.mu
        l2_prediction = layer2.mu
        l2_precision = 2.0 # Strong top-down belief
        
        # --- Step 2: L1 updates ---
        # L1 sees 'val' and hears 'l2_prediction'
        l1_state = layer1.step(val, prec, 
                               top_down_prediction=l2_prediction, 
                               top_down_precision=l2_precision)
        
        # --- Step 3: L2 updates ---
        # L2 sees L1's state as "Input".
        # L2 has no parent, so no top-down.
        # L2 treats L1 as its sensory data.
        # Precision of L2's input depends on L1's confidence?
        # For now, constant precision.
        layer2.step(l1_state[0], input_precision=1.0) 
        # Note: we feed l1_state[0] (value) to L2. 
        # L2 will estimate derivatives of L1 internally.
        
        l1_history.append(l1_state[0])
        l2_history.append(layer2.mu[0])
        
    l1_history = np.array(l1_history)
    l2_history = np.array(l2_history)
    
    print("Plotting...")
    plt.figure(figsize=(12, 6))
    
    plt.plot(t, input_signal, 'k--', label='Input', alpha=0.3)
    plt.plot(t, l1_history, 'r-', label='Layer 1 (Sensory)', linewidth=1.5)
    plt.plot(t, l2_history, 'b-', label='Layer 2 (Context)', linewidth=1.5, alpha=0.7)
    
    plt.axvspan(8.0, 12.0, color='yellow', alpha=0.3, label='Occlusion')
    
    plt.title('Phase 3: Hierarchical Object Permanence')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('hierarchy_plot.png')
    print("Saved hierarchy_plot.png")

if __name__ == "__main__":
    run_experiment()
