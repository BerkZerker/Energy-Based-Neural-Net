import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym  # Using gymnasium as the modern standard for OpenAI Gym
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
# Target Hardware: RTX 4070 (Code is CUDA ready)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Network Architecture
INPUT_SIZE = 4  # Cart Position, Cart Velocity, Pole Angle, Pole Velocity
HIDDEN_SIZE = 256  # Wide layers as per
OUTPUT_SIZE = 2  # Force Left, Force Right

# Physics Simulation Parameters
T_RELAX = 30  # Number of steps for neurons to settle (relax phase)
EPSILON = 0.5  # Step size for neuron dynamics
BETA = 0.5  # Nudge strength (how hard reality pushes the neurons)
LEARNING_RATE = 0.01


class EqPropLayer(nn.Module):
    """
    A single layer of the Equilibrium Propagation network.
    Stores weights W and biases b.
    """

    def __init__(self, in_size, out_size):
        super().__init__()
        # Initialize weights with small random values
        self.W = nn.Parameter(torch.randn(in_size, out_size, device=DEVICE) * 0.05)
        self.b = nn.Parameter(torch.zeros(out_size, device=DEVICE))

    def forward(self, x):
        # Standard linear transformation: xW + b
        return torch.mm(x, self.W) + self.b


class EqPropRobot(nn.Module):
    """
    The Equilibrium Propagation Network adapted for RL.
    Treats the network as a physical energy system.
    """

    def __init__(self):
        super().__init__()
        self.layer1 = EqPropLayer(INPUT_SIZE, HIDDEN_SIZE)
        self.layer2 = EqPropLayer(HIDDEN_SIZE, OUTPUT_SIZE)

        # State neurons (s) are persistent states, not just transient activations
        # We initialize them to 0 but they evolve over time T
        self.s_hidden = torch.zeros(1, HIDDEN_SIZE, device=DEVICE)
        self.s_output = torch.zeros(1, OUTPUT_SIZE, device=DEVICE)

    def energy(self, x, h, y):
        """
        Calculates the Total Energy (E) of the system.
        E = 1/2 sum(s^2) - sum(sWs) - sum(bs)
        """
        # Self-energy (spring term keeping neurons close to 0)
        E_self = 0.5 * (torch.sum(h**2) + torch.sum(y**2))

        # Interaction energy (neurons stimulating each other)
        # Note: We negate because high interaction = low energy (stable state)
        inter_1 = -(torch.mm(x, self.layer1.W) + self.layer1.b) * h
        inter_2 = -(torch.mm(h, self.layer2.W) + self.layer2.b) * y

        E_interaction = torch.sum(inter_1) + torch.sum(inter_2)

        return E_self + E_interaction

    def relax(self, x, target_y=None, beta=0.0):
        """
        Physics Simulation: Neurons interact for T steps to minimize Energy.

        Args:
            x: Input state (clamped)
            target_y: The "Reality" signal (optional nudge)
            beta: Nudge strength. If 0, this is the "Free Phase".
        """
        # Clone current states to start physics simulation
        h = self.s_hidden.clone().detach().requires_grad_(True)
        y = self.s_output.clone().detach().requires_grad_(True)

        # The for loop for the physics settling
        for t in range(T_RELAX):
            # Calculate Energy
            E = self.energy(x, h, y)

            # If Nudging : Output neurons are pulled toward the correct label/action
            if target_y is not None and beta != 0:
                # Add an elastic force pulling y towards target_y: beta * ||y - target||^2
                cost = beta * torch.sum((y - target_y) ** 2)
                E = E + cost

            # Calculate gradients (forces) acting on neurons
            grads = torch.autograd.grad(E, [h, y], create_graph=False)
            grad_h, grad_y = grads

            # Update neuron states (Gradient Descent on Energy = Relaxing)
            # Clip to 0-1 to simulate biological firing rates (Hard Sigmoid)
            h.data = torch.clamp(h.data - EPSILON * grad_h, 0, 1)
            y.data = torch.clamp(y.data - EPSILON * grad_y, 0, 1)

        return h, y

    def update_weights(self, free_states, nudged_states):
        """
        Local Learning: Weights update based on difference between
        "Free" state and "Nudged" state.
        Formula: Delta_W = (1/beta) * (s_nudged * s_prev_nudged - s_free * s_prev_free)
        """
        (x, h_free, y_free) = free_states
        (x, h_nudge, y_nudge) = nudged_states

        # Calculate local learning rules
        # Layer 1 Gradients
        # Outer product of input(x) and hidden(h) difference
        dh = h_nudge - h_free
        dw1 = torch.mm(x.t(), dh) / BETA
        db1 = torch.sum(dh, dim=0) / BETA

        # Layer 2 Gradients
        dy = y_nudge - y_free
        # Note: We use h_nudge for the update to approximate the chain rule locally
        dw2 = torch.mm(h_nudge.t(), dy) / BETA
        db2 = torch.sum(dy, dim=0) / BETA

        # Apply updates (Stochastic Gradient Descent)
        with torch.no_grad():
            self.layer1.W.add_(LEARNING_RATE * dw1)
            self.layer1.b.add_(LEARNING_RATE * db1)
            self.layer2.W.add_(LEARNING_RATE * dw2)
            self.layer2.b.add_(LEARNING_RATE * db2)


# ==========================================
# HELPER: The "Reality" Signal
# ==========================================
def get_oracle_correction(state):
    """
    Since the model starts knowing nothing, we need a signal from "Reality" to nudge it.
    In RL, this is usually the reward. For this demo, we use a simple physics heuristic
    to represent the 'physical laws' correcting the robot.

    If pole leans left, Reality dictates we SHOULD have pushed left.
    """
    angle = state[2]
    # Target: [Push Left, Push Right] (One-hot-ish)
    if angle < 0:
        return torch.tensor(
            [[0.0, 1.0]], device=DEVICE
        )  # Lean Left -> Push Left (Vector [0,1])
    else:
        return torch.tensor(
            [[1.0, 0.0]], device=DEVICE
        )  # Lean Right -> Push Right (Vector [1,0])


# ==========================================
# MAIN EXECUTION: Phase 3
# ==========================================
def main():
    print("🚀 Initializing Phase 3: The 'Robot' Environment (CartPole-v1)")
    print(f"   Hardware: {DEVICE}")

    # New Dataset: OpenAI Gymnasium
    env = gym.make("CartPole-v1", render_mode="human")
    robot = EqPropRobot().to(DEVICE)

    episodes = 50

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # 1. Preprocess State
            # Normalize slightly to keep inputs roughly in -1 to 1 range
            x = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            # ----------------------------------------------
            # PHASE A: The "Free" Phase (Prediction/Action)
            # ----------------------------------------------
            # Run physics simulation with NO nudge (beta=0)
            # Neurons "relax" into equilibrium
            h_free, y_free = robot.relax(x, target_y=None, beta=0.0)

            # Decide Action: Select neuron with higher activity
            # Output Neurons: 2 values (Force Left, Force Right)
            action = torch.argmax(y_free).item()

            # ----------------------------------------------
            # PHASE B: Interaction with Environment
            # ----------------------------------------------
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # ----------------------------------------------
            # PHASE C: The "Nudge" Phase (Learning)
            # ----------------------------------------------
            # Online Learning: The "Nudge" comes from the actual next frame/reality.
            # We calculate what the "Correct" output state SHOULD have been to survive.
            target_signal = get_oracle_correction(state)

            # Run physics simulation WITH nudge (beta > 0)
            # Output neurons physically pulled toward the correct label
            h_nudge, y_nudge = robot.relax(x, target_y=target_signal, beta=BETA)

            # ----------------------------------------------
            # PHASE D: Weight Update
            # ----------------------------------------------
            # Update based on difference between Free and Nudged
            robot.update_weights((x, h_free, y_free), (x, h_nudge, y_nudge))

            # Update state for next loop
            state = next_state
            total_reward += 1

            # Dynamic Control visualization
            if done:
                print(
                    f"Episode {ep + 1}: Survived {total_reward} steps. "
                    f"Final Energy Gap: {(y_nudge - y_free).abs().mean().item():.5f}"
                )

    env.close()
    print("✅ Phase 3 Complete: Robot learned online.")


if __name__ == "__main__":
    main()
