import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

# --- Configuration ---
BATCH_SIZE = 128
LEARNING_RATE = 0.1  # EqProp often needs higher LRs than Backprop
N_EPOCHS = 5
BETA = 1.0  # Nudging strength
DT = 0.5  # Integration step size (discrete time)
T_FREE = 30  # Time steps for Free Phase settling
T_NUDGE = 10  # Time steps for Nudged Phase settling
HIDDEN_DIM = 500  # Size of hidden layer
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Running on device: {DEVICE}")

# --- Dataset Preparation ---
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x)),  # Flatten 28x28 -> 784
    ]
)

train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# --- The Equilibrium Propagation Model ---
class EqPropNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # We manually manage weights to ensure they are used symmetrically
        # W1: Input -> Hidden
        self.W1 = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.b1 = nn.Parameter(torch.Tensor(hidden_dim))

        # W2: Hidden -> Output
        self.W2 = nn.Parameter(torch.Tensor(hidden_dim, output_dim))
        self.b2 = nn.Parameter(torch.Tensor(output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        # Glorot initialization works well
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)

    def hard_sigmoid(self, x):
        """Hard Sigmoid activation (0-1 range) commonly used in EqProp papers"""
        return torch.clamp(x, 0.0, 1.0)

    def energy(self, x, h, y):
        """
        Calculates the Total Energy (E) of the system.
        E = (Sum of squared states) - (Interactions) - (Biases)
        """
        # Self-energy terms (state decay)
        E_h = 0.5 * torch.sum(h**2, dim=1)
        E_y = 0.5 * torch.sum(y**2, dim=1)

        # Interaction terms (symmetric connections)
        # Note: We use W^T for the backward flow implicitly by defining energy scalar
        E_xh = torch.sum(torch.matmul(x, self.W1) * h, dim=1)
        E_hy = torch.sum(torch.matmul(h, self.W2) * y, dim=1)

        # Bias terms
        E_b1 = torch.sum(self.b1 * h, dim=1)
        E_b2 = torch.sum(self.b2 * y, dim=1)

        # Total Energy (negative sign on interactions because we want to MINIMIZE E)
        # To maximize harmony (high interaction), we minimize negative interaction.
        return E_h + E_y - E_xh - E_hy - E_b1 - E_b2

    def cost(self, y, target):
        """Squared Error Cost function"""
        # Convert integer target to one-hot
        target_onehot = F.one_hot(target, num_classes=self.output_dim).float()
        return 0.5 * torch.sum((y - target_onehot) ** 2, dim=1)

    def step_dynamics(self, x, h, y, target=None, beta=0.0):
        """
        Performs one step of Euler integration (physics settling).
        h_new = h_old - epsilon * dF/dh
        y_new = y_old - epsilon * dF/dy
        """
        # We require gradients of Energy w.r.t states h and y
        # Enable grad for states (temporarily)
        h = h.detach().requires_grad_(True)
        y = y.detach().requires_grad_(True)

        # Calculate Energy
        E = self.energy(x, h, y)
        total_energy = torch.sum(E)

        # If Nudged Phase (beta > 0), add the cost to the energy
        if beta > 0 and target is not None:
            C = self.cost(y, target)
            total_energy += beta * torch.sum(C)

        # Compute gradients (Forces acting on neurons)
        grads = torch.autograd.grad(total_energy, [h, y], create_graph=False)
        dE_dh, dE_dy = grads[0], grads[1]

        # Update states (Discrete Euler integration)
        # We move in the OPPOSITE direction of the gradient to minimize Energy
        h_new = h - DT * dE_dh
        y_new = y - DT * dE_dy

        # Apply activation function (Clamp to 0-1 range)
        h_new = self.hard_sigmoid(h_new)
        y_new = self.hard_sigmoid(y_new)

        return h_new.detach(), y_new.detach()

    def relax(self, x, h_init, y_init, T, target=None, beta=0.0):
        """Run the network dynamics for T steps to reach equilibrium"""
        h, y = h_init, y_init
        for _ in range(T):
            h, y = self.step_dynamics(x, h, y, target, beta)
        return h, y


# --- Training Helper ---
def train(model, loader, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (x, target) in enumerate(loader):
        x, target = x.to(DEVICE), target.to(DEVICE)

        # Initialize hidden and output states (usually zeros or random)
        h = torch.zeros(x.size(0), HIDDEN_DIM).to(DEVICE)
        y = torch.zeros(x.size(0), 10).to(DEVICE)

        # --- 1. FREE PHASE (Inference) ---
        # Clamp Input (x), let h and y settle to minimize Internal Energy
        h_free, y_free = model.relax(x, h, y, T_FREE, target=None, beta=0.0)

        # Check accuracy based on Free Phase prediction
        preds = y_free.argmax(dim=1)
        correct += preds.eq(target).sum().item()
        total += x.size(0)

        # --- 2. NUDGED PHASE (Learning) ---
        # Clamp Input (x) AND Nudge Output (y) slightly toward Target
        # Start from the free state (equilibrium) to save time
        h_nudged, y_nudged = model.relax(x, h_free, y_free, T_NUDGE, target, BETA)

        # --- 3. WEIGHT UPDATE (EqProp Rule) ---
        # The gradient is approx: (1/beta) * ( dE(nudged)/dTheta - dE(free)/dTheta )
        # We use PyTorch autograd to compute dE/dTheta for us.

        optimizer.zero_grad()

        # Calculate "Free" Energy gradients
        # We act as if we want to MAXIMIZE Free Energy (subtracting it in the update)
        E_free = torch.sum(model.energy(x, h_free, y_free))
        E_free.backward()

        # Store negative gradients from free phase
        grad_free = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_free[name] = param.grad.clone()

        optimizer.zero_grad()

        # Calculate "Nudged" Energy gradients
        # We want to MINIMIZE Nudged Energy
        E_nudged = torch.sum(model.energy(x, h_nudged, y_nudged))
        E_nudged.backward()

        # Combine gradients: Grad = (1/beta) * (Grad_Nudged - Grad_Free)
        for name, param in model.named_parameters():
            if param.grad is not None:
                # The .backward() computed dE_nudged/dW.
                # We subtract dE_free/dW (which we stored in grad_free).

                # Note on signs:
                # Standard SGD does: theta = theta - lr * grad
                # We want theta to move to reduce the contrastive energy.
                # EqProp Update: Delta W ~ (Activity_Nudged - Activity_Free)

                # In PyTorch, param.grad holds dE_nudged/dW.
                # We want the final grad passed to optimizer to be:
                # (dE_nudged/dW - dE_free/dW) / beta

                param.grad = (param.grad - grad_free[name]) / BETA

        optimizer.step()

        # For logging, loss is the Squared Error of the Free Phase
        with torch.no_grad():
            batch_loss = model.cost(y_free, target).mean()
            total_loss += batch_loss.item()

    return total_loss / len(loader), 100.0 * correct / total


def evaluate(model, loader):
    model.eval()  # Not strictly necessary as we don't use dropout/norm layers, but good practice
    correct = 0
    total = 0
    for x, target in loader:
        x, target = x.to(DEVICE), target.to(DEVICE)
        h = torch.zeros(x.size(0), HIDDEN_DIM).to(DEVICE)
        y = torch.zeros(x.size(0), 10).to(DEVICE)

        # Only run Free Phase for evaluation
        _, y_free = model.relax(x, h, y, T_FREE, target=None, beta=0.0)

        preds = y_free.argmax(dim=1)
        correct += preds.eq(target).sum().item()
        total += x.size(0)
    return 100.0 * correct / total


# --- Main Execution ---
model = EqPropNetwork(input_dim=784, hidden_dim=HIDDEN_DIM, output_dim=10).to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

print(f"Starting Training for {N_EPOCHS} epochs...")
print("-" * 60)

for epoch in range(1, N_EPOCHS + 1):
    start_time = time.time()
    avg_loss, train_acc = train(model, train_loader, optimizer)
    test_acc = evaluate(model, test_loader)
    end_time = time.time()

    print(
        f"Epoch {epoch}: Loss = {avg_loss:.4f} | Train Acc = {train_acc:.2f}% | Test Acc = {test_acc:.2f}% | Time = {end_time - start_time:.1f}s"
    )

print("-" * 60)
print("Training Complete.")
