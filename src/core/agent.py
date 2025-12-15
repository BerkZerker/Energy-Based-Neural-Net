import numpy as np
from core.node import CorticalColumn

class ActiveInferenceAgent:
    def __init__(self, dt=0.01):
        self.dt = dt
        
        # The "Brain": A cortical column tracking the Target's motion.
        # It learns to predict where the target is going.
        self.brain = CorticalColumn(order=2, dt=dt, learning_rate=10.0, w_learning_rate=0.1)
        
        # The "Body": Simple 1D point mass
        # motor_state = [position, velocity]
        self.hand_pos = 0.0
        self.hand_vel = 0.0
        self.mass = 1.0
        self.friction = 2.0 # Damping
        
        # Reflex Arc Gain
        # How strongly does Prediction Error drive the motor?
        self.alpha = 50.0 

    def step(self, target_pos):
        """
        1. Sensation: See the Target.
        2. Perception: Brain updates belief about Target.
        3. Action: Brain expects Hand == Target. Error drives Hand.
        """
        
        # --- PERCEPTION ---
        # The brain tries to track the target.
        # Input to brain is the VISUAL location of the target.
        # This updates the brain's internal model (mu) of the target.
        target_belief = self.brain.step(target_pos, input_precision=1.0)
        
        # --- ACTION (The Reflex Arc) ---
        # Crucial concept in FEP:
        # The agent *predicts* that its Proprioception (Hand) matches its Exteroception (Target Belief).
        # Proprioceptive Prediction Error = (Target_Belief - Hand_Position)
        # Action minimizes this error by changing Hand_Position.
        
        # Error in position
        proprio_error = target_belief - self.hand_pos
        
        # Simple Physical Motor Dynamics
        # Force = alpha * Error (Hooke's Law / P-Controller)
        force = self.alpha * proprio_error
        
        # Physics Step (Euler)
        # a = F/m - friction*v
        acc = (force / self.mass) - (self.friction * self.hand_vel)
        
        self.hand_vel += acc * self.dt
        self.hand_pos += self.hand_vel * self.dt
        
        return self.hand_pos, target_belief
