import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np # PennyLane's NumPy handles gradients
from torch import vmap

###############################################################################
# Helpers
###############################################################################

def _build_quantum_circuit(num_qubits, num_layers_vqc):
    """Return a Torch‑compatible Pennylane QNode."""

    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        # inputs: (num_qubits,)  ‑‑ already flattened
        for i in range(num_qubits):
            qml.RY(inputs[i] * torch.pi, wires=i)
        qml.StronglyEntanglingLayers(weights, wires=range(num_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

    # Torch can treat this as a pure PyTorch function
    return circuit




# --- Quantum Device and Circuit Definition ---
# Define a quantum device (simulator). For a real quantum computer,
# you would configure a different device (e.g., qml.device("qiskit.ibmq", wires=4, backend="ibmq_qasm_simulator")).
# For demonstration, we use a local simulator.
# The number of wires (qubits) determines the capacity of your quantum layer.
NUM_QUBITS = 4
# dev = qml.device("default.qubit", wires=NUM_QUBITS)
dev = qml.device("lightning.qubit", wires=NUM_QUBITS)

@qml.qnode(dev, interface="torch") # Integrate with PyTorch's autograd
def quantum_circuit(inputs, weights):
    """
    A Variational Quantum Circuit (VQC) acting as a quantum layer.

    Args:
        inputs (torch.Tensor): Classical input features to be encoded into qubits.
                               Expected shape: (NUM_QUBITS,)
        weights (torch.Tensor): Trainable parameters for the quantum gates.
                                Shape depends on the chosen ansatz.
    Returns:
        list[float]: Expectation values of PauliZ for each qubit, serving as classical outputs.
    """
    # Ensure inputs are detached from the current graph, moved to CPU, and converted to NumPy
    # PennyLane's QNode expects NumPy arrays for input data, but it handles PyTorch tensors
    # for trainable parameters.
    inputs_np = inputs.detach().cpu().numpy()

    # Step 1: Encode classical inputs into quantum states
    # Angle Encoding: Map each input feature to a rotation angle for a qubit.
    # We normalize inputs to [0, pi] for RY rotation.
    for i in range(NUM_QUBITS):
        # Ensure the input feature exists for the current qubit
        if i < inputs_np.shape[0]:
            qml.RY(inputs_np[i] * np.pi, wires=i)
        else:
            # If input_features < NUM_QUBITS, initialize remaining qubits to |0>
            # or apply a default rotation. For simplicity, we'll assume input_features == NUM_QUBITS
            # or handle padding in the classical layer.
            pass # Qubit remains in |0> state if no input is mapped

    # Step 2: Apply a parameterized quantum layer (ansatz)
    # StronglyEntanglingLayers is a common hardware-efficient ansatz.
    # It consists of alternating layers of single-qubit rotations and entangling gates.
    # The 'weights' parameter defines the rotation angles for these gates.
    # The shape of 'weights' should match the ansatz's expectation: (num_layers, num_qubits, 3)
    qml.StronglyEntanglingLayers(weights, wires=range(NUM_QUBITS))

    # Step 3: Measure expectation values of qubits
    # We measure the expectation value of the PauliZ operator for each qubit.
    # These expectation values are classical numbers between -1 and 1,
    # which can then be fed into subsequent classical layers.
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]







# --- Hybrid Quantum-Classical PyTorch Module ---
###############################################################################
# QuantumLayer module
###############################################################################

class QuantumLayer(nn.Module):
    """Maps classical features → quantum circuit → classical features."""

    def __init__(self, input_features: int, output_features: int,
                 num_qubits: int = 4, num_layers_vqc: int = 2):
        super().__init__()
        self.num_qubits = num_qubits
        self.input_features = input_features
        self.output_features = output_features

        # 1) Linear projection to qubit count
        self.pre_projection = nn.Linear(input_features, num_qubits)
        # alias for backward‑compat
        self.pre_classical = self.pre_projection

        # 2) Trainable VQC weights
        self.q_weights = nn.Parameter(torch.randn((num_layers_vqc, num_qubits, 3)))

        # 3) Build (batched) QNode
        base_qnode = _build_quantum_circuit(num_qubits, num_layers_vqc)
        self.qnode_batched = self._make_batched_qnode(base_qnode)

        # 4) Project back to output size
        self.post_projection = nn.Linear(num_qubits, output_features)
        self.post_classical = self.post_projection  # alias

    # ------------------------------------------------------------------
    @staticmethod
    def _make_batched_qnode(base_qnode):
        """Return a function f(inputs, weights) that is batched over inputs."""
        try:
            return torch.vmap(base_qnode, in_dims=(0, None))  # torch ≥2.0
        except Exception:
            try:
                from torch.func import vmap  # torch <2.0
                return vmap(base_qnode, in_dims=(0, None))
            except Exception as e:
                print("[QuantumLayer] vmap unavailable – falling back to loop:", e)
                return lambda x, w: torch.stack([base_qnode(xi, w) for xi in x], 0)

    # ------------------------------------------------------------------
    def forward(self, x):
        N, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(-1, C)          # (N*H*W, C)
        x_q = self.pre_projection(x)                      # (N*H*W, num_qubits)

        # ── quantum ───────────────────────────────────────────────
        q_out = self.qnode_batched(x_q.float(), self.q_weights)

        # 1) 리스트 → Tensor
        if isinstance(q_out, (list, tuple)):
            q_out = torch.stack(q_out, dim=1)             # (N*H*W, num_qubits)

        # 2) dtype 맞추기 (float64 → float32)
        q_out = q_out.to(dtype=self.post_projection.weight.dtype)

        # ── back to classical ─────────────────────────────────────
        x_c = self.post_projection(q_out)                 # (N*H*W, out_feat)
        x_c = x_c.reshape(N, H, W, self.output_features).permute(0, 3, 1, 2)
        return x_c
