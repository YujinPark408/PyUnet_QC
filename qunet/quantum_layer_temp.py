import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np
from torch import vmap

# --- Quantum Device and Circuit Definition ---
NUM_QUBITS = 4  # pre_q_out 형상에 맞게 복원
dev = qml.device("lightning.qubit", wires=NUM_QUBITS)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    """
    A Variational Quantum Circuit (VQC) acting as a quantum layer.

    Args:
        inputs (torch.Tensor): Classical input features, expected shape: (NUM_QUBITS,)
        weights (torch.Tensor): Trainable parameters, shape: (num_layers_vqc, num_qubits, 3)
    Returns:
        torch.Tensor: Expectation values of PauliZ.
    """
    if inputs.dim() != 1 or inputs.size(0) < NUM_QUBITS:
        raise ValueError(f"Expected 1D tensor with at least {NUM_QUBITS} elements, got {inputs.shape}")
    inputs = inputs.float()
    inputs_np = inputs[:NUM_QUBITS].detach().cpu().numpy()  # NUM_QUBITS까지만 사용

    for i in range(NUM_QUBITS):
        if i < inputs_np.shape[0]:
            qml.RY(inputs_np[i] * np.pi, wires=i)
        else:
            pass  # 남은 쿼빗은 |0> 상태

    qml.StronglyEntanglingLayers(weights, wires=range(NUM_QUBITS))
    return torch.tensor([qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)], device=inputs.device)

class QuantumLayer(nn.Module):
    """
    A PyTorch module incorporating a Variational Quantum Circuit (VQC).
    """
    def __init__(self, input_features, output_features, num_qubits=NUM_QUBITS, num_layers_vqc=3):
        super(QuantumLayer, self).__init__()
        self.pre_quantum_classical_layer = nn.Linear(input_features, num_qubits)
        weight_shape = (num_layers_vqc, num_qubits, 3)
        self.q_weights = nn.Parameter(torch.rand(weight_shape, requires_grad=True))
        self.quantum_node = quantum_circuit
        self.post_quantum_classical_layer = nn.Linear(num_qubits, output_features)

    def forward(self, x):
        original_shape = x.shape
        batch_size = original_shape[0]

        if len(original_shape) > 2:
            channels = original_shape[1]
            H = original_shape[2]
            W = original_shape[3]
            x_reshaped_for_quantum = x.permute(0, 2, 3, 1).reshape(-1, channels)
        else:
            x_reshaped_for_quantum = x
            channels = original_shape[1]

        pre_q_out = self.pre_quantum_classical_layer(x_reshaped_for_quantum)
        print(f"pre_q_out shape: {pre_q_out.shape}")  # 디버깅

        try:
            quantum_outputs = vmap(lambda x: self.quantum_node(x, self.q_weights))(pre_q_out)
        except Exception as e:
            print(f"Quantum error: {e}, pre_q_out sample: {pre_q_out[0] if pre_q_out.numel() > 0 else 'empty'}")
            raise

        post_q_out = self.post_quantum_classical_layer(quantum_outputs)

        if len(original_shape) > 2:
            final_output = post_q_out.reshape(batch_size, H, W, -1).permute(0, 3, 1, 2)
        else:
            final_output = post_q_out

        return final_output