import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from core.agent import ActiveInferenceAgent

def run_experiment():
    dt = 0.01
    duration = 20.0
    time_steps = int(duration / dt)
    t = np.linspace(0, duration, time_steps)
    
    # Target: A complex path (Sine wave + slow drift)
    # This tests if the brain can track composite motion
    target_pos = np.sin(1.0 * t) + 0.5 * np.cos(0.5 * t)
    
    agent = ActiveInferenceAgent(dt=dt)
    
    hand_positions = []
    brain_beliefs = []
    
    print("Starting Phase 2: The Tracker...")
    for i in range(time_steps):
        target = target_pos[i]
        
        hand, belief = agent.step(target)
        
        hand_positions.append(hand)
        brain_beliefs.append(belief)
        
    hand_positions = np.array(hand_positions)
    brain_beliefs = np.array(brain_beliefs)
    
    print("Plotting results...")
    plt.figure(figsize=(12, 6))
    
    plt.plot(t, target_pos, 'k--', label='Target (External World)', alpha=0.6)
    plt.plot(t, brain_beliefs, 'b:', label='Brain Belief (Internal)', alpha=0.8)
    plt.plot(t, hand_positions, 'r-', label='Hand Position (Action)', linewidth=2)
    
    plt.title('Phase 2: Active Inference Tracker')
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.legend()
    plt.grid(True)
    
    output_path = 'tracker_plot.png'
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    run_experiment()
