import numpy as np


class ActiveInferenceAgent:
    def __init__(self, dt=0.01):
        self.dt = dt

        # --- 1. INTERNAL STATE (The Brain) ---
        # 0: Position, 1: Velocity
        self.mu = np.array([0.0, 0.0])

        # --- 2. DOPAMINE (Precision) ---
        # High precision = trusting sensory data
        self.pi_s = 5.0  # Sensory Precision (Confidence in eyes)
        self.pi_w = 1.0  # Dynamics Precision (Confidence in internal physics)

        # --- 3. ACTION (The Body) ---
        self.action = 0.0  # This modifies the external world position

    def step(self, raw_sensory_input):
        """
        raw_sensory_input: The true position of the target in the world.
        """

        # --- STEP A: SENSATION & ACTION ---
        # The agent's 'action' shifts its viewpoint.
        # e.g., If I look right, the object moves left in my visual field.
        sensory_observation = raw_sensory_input - self.action

        # --- STEP B: PREDICTION (Top-Down) ---
        # Predict current position based on previous state + velocity
        # D_mu is the 'Generalized Motion' (Velocity predicts Position change)
        pred_obs = self.mu[0] + (self.mu[1] * self.dt)

        # --- STEP C: PREDICTION ERRORS (Weighted by Dopamine) ---
        # Sensory Error: (Observation - Prediction) * Sensory_Precision
        eps_s = self.pi_s * (sensory_observation - pred_obs)

        # Dynamics Error: (Velocity - Expected_Velocity) * Dynamics_Precision
        # We expect velocity to stay constant (Newton's 1st law as prior)
        eps_w = self.pi_w * (self.mu[1] - 0.0)

        # --- STEP D: PERCEPTION (Update Internal State) ---
        # Minimizing error by changing internal beliefs (mu)
        # dF/dmu terms...

        # Update Position Belief
        self.mu[0] += self.dt * (self.mu[1] + eps_s)

        # Update Velocity Belief
        self.mu[1] += self.dt * (eps_w - eps_s)  # Velocity corrects position error

        # --- STEP E: ACTION (Update Output) ---
        # Minimizing error by changing external world (action)
        # Gradient descent on the sensory error.
        # If Error is positive (Object is to my right), Action increases (Turn right)

        da = -1.0 * eps_s  # Simple reflex arc
        self.action += self.dt * da

        return self.action, self.mu[0]


def run_demo(steps=300, dt=0.01, target_position=1.0):
    agent = ActiveInferenceAgent(dt=dt)
    history = {"action": [], "belief_pos": [], "target": []}

    for _ in range(steps):
        a, pos = agent.step(target_position)
        history["action"].append(a)
        history["belief_pos"].append(pos)
        history["target"].append(target_position)

    return history


def main():
    hist = run_demo()
    print("Run complete")
    print(f"Final action: {hist['action'][-1]:.4f}")
    print(f"Final belief position: {hist['belief_pos'][-1]:.4f}")


if __name__ == "__main__":
    main()
