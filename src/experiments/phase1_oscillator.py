import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from core.node import CorticalColumn

def run_experiment():
    dt = 0.005 # Smaller timestep for stability
    duration = 15.0 # Longer duration to allow learning
    time_steps = int(duration / dt)
    t = np.linspace(0, duration, time_steps)
    
    frequency = 1.0 
    input_signal = np.sin(2 * np.pi * frequency * t)
    
    # Mask input between t=10 and t=12 (The Cut)
    # We give it 10 seconds to learn first!
    input_precision = np.ones_like(t)
    cut_start_idx = int(10.0 / dt)
    cut_end_idx = int(12.0 / dt)
    input_precision[cut_start_idx:cut_end_idx] = 0.0
    
    # Increase learning rates
    agent = CorticalColumn(order=2, dt=dt, learning_rate=10.0, w_learning_rate=0.5)
    
    predictions = []
    
    print("Starting Simulation with Learning...")
    for i in range(time_steps):
        val = input_signal[i]
        prec = input_precision[i]
        
        pred = agent.step(val, prec)
        predictions.append(pred)
        
    predictions = np.array(predictions)
    
    print("Plotting results...")
    plt.figure(figsize=(12, 6))
    plt.plot(t, input_signal, 'k--', label='True Input', alpha=0.5)
    plt.plot(t, predictions, 'r-', label='Agent Perception', linewidth=2)
    plt.axvspan(10.0, 12.0, color='yellow', alpha=0.3, label='Blind')
    
    # Limit y-axis to avoid showing explosions if they happen
    plt.ylim(-2.5, 2.5)
    
    plt.title(f'Phase 1: Oscillator with Hebbian Learning\nFreq={frequency}Hz')
    plt.legend()
    plt.grid(True)
    
    output_path = 'oscillator_plot_v2.png'
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    run_experiment()