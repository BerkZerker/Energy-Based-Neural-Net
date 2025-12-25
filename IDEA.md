# Idea for a Neural Network Architecture

The idea is that each neuron cell has it's connections, and it's state, and it's prediction, and local error and so on and so forth, but it only handles it's own local data. It also has a "direction" vector that will function as a sort of memory of what the previous neuron predictions were so that it is resistant to catastrophic forgetting.

The connections between neurons are not static, they are dynamic and can change based on the local error and the direction vector - really whatever local rule I come up with. This allows the network to adapt to new information while still retaining knowledge from previous training, all without the need for a layer-based structure.
