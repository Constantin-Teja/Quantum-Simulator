import numpy as np
from Quantum_Helper import Quantum_Helper as qh
from Gate import Gate

class State:

    def __init__(self, n_qubits: np.uint16):
        self.n_qubits = n_qubits # cteja TODO: make private
        self.state = np.zeros((2**n_qubits, 1), dtype=np.complex64)
        self.state[0] = 1 # Set |00...0> as default state

    def initialize_state(self, qubit_values):
        assert len(qubit_values) == self.n_qubits
        self.state = np.zeros((2**len(qubit_values), 1), dtype=np.complex64)
        sum_of_coefficients = 0
        for i in range(len(qubit_values)):
            sum_of_coefficients = sum_of_coefficients + qubit_values[i] * 2**(len(qubit_values) - i - 1)
        self.state[sum_of_coefficients][0] = 1

    def apply_gate(self, gate, n_gate, starting_qubit):
        applied_gate = np.identity(1)

        for i in range(starting_qubit):
            applied_gate = np.kron(applied_gate, np.identity(2))
        
        applied_gate = np.kron(applied_gate, gate.get_gate())

        for i in range(starting_qubit + n_gate, self.n_qubits):
            applied_gate = np.kron(applied_gate, np.identity(2))

        self.state = np.dot(applied_gate, self.state)
       
    def apply_H_gate(self, target_qubit):
        self.apply_gate(Gate.gates["H"], 1, target_qubit)
    
    def apply_X_gate(self, target_qubit):
        self.apply_gate(Gate.gates["X"], 1, target_qubit)
    
    # TODO: unittest
    def apply_S_gate(self, target_qubit):
        self.apply_gate(Gate.gates["S"], 1, target_qubit)

    # TODO: unittest
    def apply_CNOT_gate(self, target_qubit):
        self.apply_gate(Gate.gates["CNOT"], 2, target_qubit)
    
    # TODO: unittest
    def apply_CH_gate(self, target_qubit):
        self.apply_gate(Gate.gates["CH"], 2, target_qubit)
    
    # TODO: unittest
    def apply_SWAP_gate(self, target_qubit):
        self.apply_gate(Gate.gates["SWAP"], 2, target_qubit)
    
    # TODO: unittest
    def apply_CNOT10_gate(self, target_qubit):
        self.apply_gate(Gate.gates["CNOT10"], 2, target_qubit)
    
    # TODO: unittest
    def apply_TOFFOLI_gate(self, target_qubit):
        self.apply_gate(Gate.gates["TOFFOLI"], 3, target_qubit)

    # TODO: unittest
    def produce_measurement(self, base_name):
        return qh.produce_measurement(self.state, self.n_qubits, base_name)

    # TODO: unittest
    def calculate_state_amplitudes(self):
        # Check if the input state vector is normalized (sum of squared amplitudes is close to 1)
        norm = np.linalg.norm(self.state)
        if not np.isclose(norm, 1.0):
            raise ValueError("Input state vector must be normalized.")

        n_qubits = int(np.log2(len(self.state)))

        # Initialize a dictionary to store the amplitudes of all basis states
        state_amplitudes = {}

        # Iterate through all possible basis states and calculate their amplitudes
        for i in range(2 ** n_qubits):
            basis_state = np.binary_repr(i, width=n_qubits)  # Convert the integer to binary representation
            amplitude = self.state[i]
            state_amplitudes[basis_state] = amplitude

        return state_amplitudes
    
    # TODO: unittest
    # Same as above but returns an array of values without the keys
    def calculate_state_amplitudes_arr(self):
        norm = np.linalg.norm(self.state)
        if not np.isclose(norm, 1.0):
            raise ValueError("Input state vector must be normalized.")
        n_qubits = int(np.log2(len(self.state)))
        state_amplitudes = []
        for i in range(2 ** n_qubits):
            amplitude = self.state[i]
            state_amplitudes.append(amplitude)

        return state_amplitudes
    
    # TODO: unittest
    def display_state_amplitudes(self):
        state_amplitudes = self.calculate_state_amplitudes()
        # Display the state amplitudes
        for basis_state, amplitude in state_amplitudes.items():
            print(f"|{basis_state}>: {amplitude}")

    def compute_density_matrix(self, base_name = "COMPUTATIONAL"):
        probabilities = qh.compute_probabilities(self.state, self.n_qubits, base_name)
        extended_base = qh.get_extended_basis("COMPUTATIONAL", self.n_qubits)

        density_matrix = np.zeros((len(self.state), len(self.state)))
        for i in range(len(extended_base)):
            density_matrix = density_matrix + probabilities[i] * np.outer(qh.compute_adjoint(extended_base[i]), extended_base[i])

        return density_matrix