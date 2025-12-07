# **Equilibrium Propagation: The "Living" Neural Network**

## **Overview**

This project implements **Equilibrium Propagation (EqProp)**, a biologically plausible alternative to Backpropagation. Unlike standard AI that calculates error derivatives globally, this model treats the neural network as a physical energy system. Neurons "relax" into equilibrium states, and learning occurs by measuring the local difference between a "Free" state (prediction) and a "Nudged" state (reality).

**Current Status:** Working implementation on the MNIST dataset using PyTorch as a physics engine.

## **The Basics (Current Script)**

### **eq_prop_mnist.py**

This script trains a fully connected network to classify handwritten digits (0-9).

- Physics Simulation: Instead of a forward() pass, the network has a relax() phase where neurons interact for T time steps to minimize the Energy function:  
  $$E = \frac{1}{2}\sum s^2 - \sum (s_i W_{ij} s_j) - \sum b s$$
- **The "Nudge":** During training, output neurons are physically pulled toward the correct label.
- **Local Learning:** Weights update based on the difference in local neuron activity between the "Free" phase and the "Nudged" phase.

### **Usage**

1. Install dependencies:  
   pip install torch torchvision matplotlib

2. Run the training script:  
   python eq_prop_mnist.py

3. Monitor the Loss (Free Energy) and Accuracy in the console.

## **Project Roadmap: "The Ghost in the Machine"**

The ultimate goal is to evolve this static classifier into a **Self-Learning Generative Agent**—a "robot" in a digital environment that learns online without catastrophic forgetting.

### **Phase 1: The "Dreaming" Model (Generative Replay)**

**Goal:** Prevent catastrophic forgetting by making the model generate its own training data.

- **Concept:** Standard classifiers map Image \-\> Label. EqProp is bidirectional. We can clamp a Label and run the physics backward to generate an Image.
- **Implementation Plan:**
  - Add a dream(target_label) function to the class.
  - Clamp the output layer to a specific number (e.g., "5").
  - Relax the network.
  - Visualize the input layer (pixels) to see if it hallucinates a "5".
- **Why:** When learning a new task, the model will "replay" these dreams to maintain old memories.

### **Phase 2: The "Global Signal" (Simulated Dopamine)**

**Goal:** Implement online learning where the model decides _when_ to learn.

- **Concept:** Currently, we force the model to learn on every batch. A biological brain only learns when it is **surprised**.
- **Implementation Plan:**
  - Create a "Global Prediction Error" metric (the magnitude of the energy jar during the Nudge).
  - **The Modulator:** If Energy \< Threshold (Prediction was good), set Learning Rate to 0\.
  - If Energy \> Threshold (Surprise\!), spike the Learning Rate.
- **Benefit:** This saves computation and prevents overwriting good memories with noise.

### **Phase 3: The "Robot" Environment (Switching to Physics)**

**Goal:** Move from static images (MNIST) to dynamic control.

- **New Dataset:** OpenAI Gymnasium (e.g., CartPole-v1 or Pendulum-v1).
- **The Architecture Change:**
  - **Input Neurons:** 4 continuous values (Cart Position, Cart Velocity, Pole Angle, Pole Velocity).
  - **Output Neurons:** 2 values (Force Left, Force Right).
- **The Task:** The network must predict the _next state_ of the physics engine.
- **Online Learning:** The "Nudge" comes from the actual next frame of the simulation. If the model predicted the pole would fall left, but it fell right, the "Reality" nudges the neurons, and the weights update instantly.

### **Phase 4: Active Inference (Planning)**

**Goal:** The agent "thinks" before it acts.

- **Concept:** Before sending a motor command, the robot runs a "Free Phase" simulation internally.
- **Loop:**
  1. Robot imagines Action A \-\> Network predicts Result A.
  2. Robot checks Energy of Result A (Is the pole upright?).
  3. If Energy is high (Bad outcome), Robot imagines Action B.
  4. Robot executes the action with the lowest predicted Energy.

## **🛠 Hardware & Optimization**

**Target Hardware:** NVIDIA RTX 4070

- **VRAM Usage:** EqProp stores "states" rather than massive activation graphs. A 4070 can handle very wide layers (width 2048+) or deeper recurrence (T=100) easily.
- **Performance Bottleneck:** The for loop in Python for the relax() phase is slow.
- **Future Optimization:** Write the step_dynamics function in a **Custom CUDA Kernel** (using Triton or raw CUDA) to run the physics settling 100x faster than standard PyTorch loops.

## **References & Inspiration**

1. _Equilibrium Propagation: Bridging the Gap Between Energy-Based Models and Backpropagation_ (Scellier & Bengio).
2. _The Free Energy Principle_ (Karl Friston).
3. _Predictive Coding Networks_ (Rao & Ballard).
