import numpy as np

class State:

    def __init__(self, n_qubits):
        # TODO task 1.1
        self.n_qubits = n_qubits
        self.state = np.zeros(2**n_qubits, dtype=np.complex64)
        self.state[0] = 1

    def initialize_state(self, qubit_values):
        assert len(qubit_values) == self.n_qubits
        # TODO task 1.2
        self.state = np.zeros(2**len(qubit_values), dtype=np.complex64)
        sumOfCoefficients = 0
        for i in range(len(qubit_values)):
            sumOfCoefficients = sumOfCoefficients + qubit_values[i] * 2**(len(qubit_values) - i - 1)
        self.state[sumOfCoefficients] = 1
    
    
    def compute_adjoint(self, matrix):
        matrix = matrix.conjugate()
        matrix = matrix.transpose()
        return matrix

    def is_valid_gate(self, gate):
        adj =  self.compute_adjoint(gate)
        if np.allclose(np.dot(adj,gate), np.eye(int(np.sqrt(gate.size)))) :
            return True
        return False
    

    def create_custom_gate(self, matrix=None, gates=None):
            """Create a custom gate, either from a matrix or by composing other gates."""
            if matrix is not None:
                if self.is_valid_gate(matrix):
                    return matrix
                else:
                    raise ValueError("Matrix is not unitary.")
            elif gates is not None:
                composite_matrix = gates[0]
                for gate in gates[1:]:
                    composite_matrix = np.dot(composite_matrix, gate)
                if self.is_valid_gate(composite_matrix):
                    return composite_matrix
                else:
                    raise ValueError("Composite gate is not unitary.")
            else:
                raise ValueError("Either a matrix or a list of gates is required.")




    def apply_gate(self, gate, n_gate, starting_qubit):
        # TODO task 1.3
        applied_gate = np.identity(1)

        for i in range(starting_qubit):
            applied_gate = np.kron(applied_gate, np.identity(2))
        
        applied_gate = np.kron(applied_gate, gate)

        for i in range(starting_qubit + n_gate, self.n_qubits):
            applied_gate = np.kron(applied_gate, np.identity(2))

        self.state = np.dot(self.state, applied_gate)
       
    def apply_H_gate(self, target_qubit):
        # TODO task 1.4
        self.apply_gate(np.array([[1, 1], [1, -1]])/np.sqrt(2), 1, target_qubit)
    
    def apply_X_gate(self, target_qubit):
        # TODO task 1.4
        self.apply_gate(np.array([[0, 1], [1, 0]]), 1, target_qubit)
    
    def apply_S_gate(self, target_qubit):
        # TODO task 1.4
        self.apply_gate(np.array([[1, 0], [0, 1j]]), 1, target_qubit)

    def apply_CNOT_gate(self, target_qubit):
        # TODO task 1.5
        self.apply_gate(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]), 2, target_qubit)
    
    def apply_CH_gate(self, target_qubit):
        # TODO task 1.5
        unuPeRadacinaDinDoi = 1 / np.sqrt(2)
        self.apply_gate(np.array([[1, 0, 0, 0], [0, unuPeRadacinaDinDoi, 0, unuPeRadacinaDinDoi], [0, 0, 1, 0], [0, unuPeRadacinaDinDoi, 0, unuPeRadacinaDinDoi]]), 2, target_qubit)
    
    def apply_SWAP_gate(self, target_qubit):
        # TODO task 1.5
        self.apply_gate(np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0 ,0], [0, 0, 0, 1]]), 2, target_qubit)
    
    def apply_CNOT10_gate(self, target_qubit):
        # TODO task 1.5
        self.apply_gate(np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1 ,0], [0, 0, 0, 1]]), 2, target_qubit)
    
    def apply_TOFFOLI_gate(self, target_qubit):
        # TODO task 1.5
        self.apply_gate(self, np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1 ,0 ,0 ,0 ,0 ,0 ,0], [0, 0, 1, 0, 0, 0, 0, 0], [0 ,0 ,0 ,1 ,0 ,0 ,0 ,0], [0 ,0 ,0 ,0 ,1 ,0 ,0 ,0], [0 ,0 ,0 ,0 ,0 ,1 ,0 ,0], [0 ,0 ,0 ,0 ,0 ,0 ,0 ,1], [0 ,0 ,0 ,0 ,0 ,0 ,1 ,0]]), 3, target_qubit)
    
    def produce_measurement(self):
        # TODO task 1.6
        probabilities = np.abs(self.state)**2

        # Randomly choose a state according to these probabilities
        result = np.random.choice(len(self.state), p=probabilities)

        # Convert the result to binary and pad with zeros to match the number of qubits
        result_binary = format(result, '0' + str(self.n_qubits) + 'b')

        # Convert the binary string to a list of integers
        result_list = [int(bit) for bit in result_binary]
        
        return result_list
    

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
    
    def display_state_amplitudes(self):
        state_amplitudes = self.calculate_state_amplitudes()
        # Display the state amplitudes
        for basis_state, amplitude in state_amplitudes.items():
            print(f"|{basis_state}>: {amplitude}")
