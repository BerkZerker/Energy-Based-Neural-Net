# Design Specification: Biologically Plausible Predictive Coding (BP-PC) Engine

## 1. Executive Summary

This document outlines the architecture for a "Biologically Plausible" neural network based on the **Free Energy Principle (FEP)**. Unlike standard Deep Learning (which relies on global loss functions and backpropagation), this system relies on **Active Inference**, **Local Hebbian Learning**, and **Asynchronous Event-Driven (Spiking) Dynamics**. The goal is to create an agent that does not just classify data, but actively simulates and regulates its environment to minimize "Surprise" (Entropy).

## 2. Core Philosophy

The agent operates on a single imperative: **Minimize Free Energy ($F$)**.
Free Energy is defined as the upper bound on "Surprise" (mismatch between expectation and reality).

$$F \approx \text{Prediction Error} + \text{Complexity Costרוי$$}

- **Perception:** The agent changes its internal state ($\mu$) to match the external world.
- **Action:** The agent changes the external world ($Input$) to match its internal state ($\mu$).
- **Learning:** The agent changes its structure ($W$) to prevent future errors.

## 3. The Node Architecture: "Generalized Cortical Column"

The fundamental unit is not a neuron, but a **Cortical Column** representing a specific physical variable (e.g., "Hand Position" or "Target Velocity").

### A. Generalized Coordinates of Motion

To solve the "Time" problem biologically, every node maintains a vector of state derivatives. It does not just know where it is; it knows where it is going.

**State Vector ($\tilde{\mu}$):**

$$\tilde{\mu} = [\mu^{(0)}, \mu^{(1)}, \mu^{(2)}]$$

- $\mu^{(0)}$: Value (Position)
- $\mu^{(1)}$: Velocity (Rate of Change)
- $\mu^{(2)}$: Acceleration (Change of Rate)

### B. The Differential Equation (Update Rule)

The state $\tilde{\mu}$ evolves continuously over time based on two opposing forces:

1. **System Dynamics ($D$):** The inertial prediction (Velocity pushes Position).
2. **Sensory Correction ($-\partial F$):** The error signal pulling the state toward reality.

$$\dot{\tilde{\mu}} = D\tilde{\mu} - \frac{\partial F}{\partial \tilde{\mu}}$$

## 4. Network Topology: Asynchronous Graph

The system is a **Directed Cyclic Graph**, not a layer-cake.

### A. Sparse Connectivity

- **No Dense Matrices:** Use an Adjacency List (Graph) structure.
- **Parents:** Nodes providing Top-Down predictions (Context).
- **Children:** Nodes providing Bottom-Up errors (Sensation).
- **Lateral:** Nodes providing inhibition/competition (Sparse Coding).

### B. Event-Driven Communication ("Spiking")

To emulate biological efficiency and asynchrony:

- **Sleep:** If Prediction Error $\approx 0$, the node is idle.
- **Integration:** Nodes continuously integrate incoming signals into their "Membrane Potential."
- **Fire:** If $|Error| > \text{Threshold}$, the Error Unit "spikes" (broadcasts its value).
- **Wake:** Receiving a spike triggers a neighbor to wake up and re-compute its gradient.

## 5. Neuromodulation: "The Chemical Layer"

We introduce global variables that modulate how the math runs.

### A. Dopamine (Precision $\Pi$)

Dopamine represents the **Confidence** or **Inverse Variance** of a signal. It acts as a "Gain Control" on Error Units.

$$\xi = \Pi \cdot (\text{Input} - \text{Prediction})$$

- **Low Dopamine:** Ignore sensory data (Dreaming / Hallucination).
- **High Dopamine:** Amplify sensory data (Attention / Panic).
- **Dynamic Precision:** The agent learns to lower $\Pi$ for noisy sensors (fog) and raise $\Pi$ for reliable ones.

## 6. Plasticity Mechanisms (Learning)

Learning occurs on three distinct timescales.

### Level 1: Inference (Fastest) - ms

Changing Activity ($\mu$) to minimize error immediately.

### Level 2: Synaptic Plasticity (Medium) - seconds/minutes

**Hebbian Learning:** Weights ($W$) update to minimize the correlation between Pre-synaptic State and Post-synaptic Error.

$$\Delta W_{ij} = \eta \cdot \text{State}_j \cdot \text{Error}_iרוי$$

- **Local Rule:** No global backprop. A synapse only knows the two nodes it touches.

### Level 3: Structural Plasticity (Slowest) - hours/days

The graph itself evolves. This is "Gardening" the brain.

#### A. Synaptic Pruning (Forgetting)

To prevent overfitting and reduce compute cost.

- **Mechanism:** Every synapse has a "Stability Score" (or Age).
- **Rule:** If $|W_{ij}| < \text{Threshold}$ AND Age > Maturity_Time:
  - **DELETE EDGE $(i, j)$**
- **Result:** The network naturally sparsifies, keeping only causal pathways.

#### B. Synaptogenesis (Growing)

To learn new concepts that current topology cannot explain.

- **Mechanism:** Monitor the Free Energy (Stress) of a node over a time window.
- **Rule:** If $\int F(t) dt > \text{Stress_Limit}$:
  - **Search:** Identify a nearby node with high activity.
  - **Grow:** Create a new random connection $W_{new}$ to that node.
  - **Split (Optional):** If stress remains critical, undergo "Mitosis" (split the node into two to increase capacity).

## 7. Action & Behavior: Active Inference

The agent does not have a separate "Controller." Motor commands are just sensory predictions that haven't come true yet.

### The Reflex Arc

1. **Prediction:** Brain predicts Hand_Position = Cup_Location.
2. **Sensation:** Eyes see Hand_Position = Table_Location.
3. **Error:** High Prediction Error!
4. **Action:** Instead of changing the belief (admitting defeat), the agent's reflex arc fires to change the input.

$$\dot{a} = - \frac{\partial F}{\partial a}$$רוי

(Move the muscle until Sensation == Prediction).

## 8. Master Algorithm (Pseudo-Code)

```python
class BioAgent:
    def __init__(self):
        self.graph = SparseGraph()
        self.dopamine_level = 1.0  # Tonic Dopamine

    def run_cycle(self, sensory_environment):
        """
        One cycle of biological simulation.
        """

        # --- 1. SENSORY INTAKE ---
        input_data = sensory_environment.get_data()
        self.graph.clamp_sensors(input_data)

        # --- 2. FAST INFERENCE (Perception) ---
        # Asynchronous "Ripple" of updates
        settled = False
        step = 0
        while not settled and step < MAX_STEPS:
            # Only update nodes triggered by events (Spikes)
            active_nodes = self.graph.get_active_nodes()

            for node in active_nodes:
                # A. Dynamics (Momentum)
                prior = node.predict_motion()

                # B. Error Calculation (with Dopamine)
                error = node.compute_error(prior, self.dopamine_level)

                # C. State Update (Descent on Free Energy)
                node.update_state(error)

                # D. Spiking (Event Generation)
                if abs(error) > SPIKE_THRESHOLD:
                    node.broadcast_spike()

            step += 1

        # --- 3. ACTIVE INFERENCE (Action) ---
        # Motor nodes reduce error by moving the "body"
        motor_actions = self.graph.get_motor_gradients()
        sensory_environment.apply_force(motor_actions)

        # --- 4. SLOW LEARNING (Plasticity) ---
        # Hebbian Weight Updates
        self.graph.update_weights_hebbian()

        # --- 5. STRUCTURAL PLASTICITY (Gardening) ---
        # Occurs rarely (e.g., every 1000 cycles)
        if self.is_sleep_cycle():
            self.prune_weak_synapses()
            self.grow_new_synapses_based_on_stress()
```

## 9. Implementation Roadmap

### Phase 1: The Oscillator (Proof of Concept)

- **Goal:** Build a single Cortical Column that tracks a Sine Wave input.
- **Success Metric:** The node establishes "Resonance" (predicts the sine wave even if input cuts out briefly).
- **Tech:** Python, NumPy.

### Phase 2: The Tracker (Active Inference)

- **Goal:** Build an agent that moves a cursor to follow a target.
- **Components:** Sensory Node, Motor Node, Reflex Arc.
- **Tech:** Simple 1D Matplotlib animation.

### Phase 3: The Hierarchy (Deep Temporal Model)

- **Goal:** Stack columns (L1 tracks Position, L2 tracks Velocity).
- **Test:** Show it a complex pattern (e.g., a bird flying behind a tree).
- **Success Metric:** Object Permanence (L2 predicts emergence from behind the tree).

### Phase 4: The Gardener (Structural Plasticity)

- **Goal:** Start with a disconnected graph.
- **Task:** Feed it data.
- **Success Metric:** Watch it "grow" the connections required to solve the task, and prune the rest.
