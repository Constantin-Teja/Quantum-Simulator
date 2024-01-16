import numpy as np

# cteja TODO: complete
# Static class
class Quantum_Helper:
    __qubit_basis = {
        # Each base vectors are represented on lines
        "COMPUTATIONAL" : np.array([ [1, 0], [0, 1] ]),
        "HADAMARD" : (1/np.sqrt(2)) * np.array([ [1, -1], [1, 1] ])
    }

    # Sqrt(2) is used multiple times and is heavy
    sqrt2 = np.sqrt(2)
    
    # TODO: unittest
    @staticmethod
    def get_extended_basis(base_name, n_qubits: np.uint64):
        basis = Quantum_Helper.__qubit_basis[base_name]
        extended_basis = basis
        for i in range(n_qubits - 1):
            extended_basis = np.kron(extended_basis, basis)
            
        return extended_basis
    
    # TODO: unittest
    @staticmethod
    def compute_adjoint(matrix):
        return matrix.conjugate().transpose()
    
    #@staticmethod
    #def set_basis(state_name, basis):
        # cteja TODO
    
    @staticmethod
    def get_base(base_name: str):
        return Quantum_Helper.__qubit_basis[base_name]
    
    @staticmethod
    def __compute_measurement_matrix(pure_state):
        return np.dot(pure_state, Quantum_Helper.compute_adjoint(pure_state))

    # TODO: unittest
    @staticmethod
    def compute_probabilities(state, n_qubits, base_name = "COMPUTATIONAL"):
        # Initialize probabilities array
        probabilities = np.zeros(len(state))

        # Compute probabilities of states in base_name
        basis_measurement = Quantum_Helper.get_extended_basis(base_name, n_qubits)

        for i in range(len(basis_measurement)):
            base = np.array([basis_measurement[i]]).transpose()
            Mi = Quantum_Helper.__compute_measurement_matrix(base)
            temp = np.dot(Mi, state)
            probabilities[i] = np.dot(Quantum_Helper.compute_adjoint(state), temp).real

        return probabilities
    
    # TODO: unittest
    @staticmethod
    def produce_measurement(state, n_qubits, base_name = "COMPUTATIONAL"):
        probabilities = Quantum_Helper.compute_probabilities(state, n_qubits, base_name)
        
        # Randomly choose a state in the specified base according to these probabilities
        measured_state_index = np.random.choice(len(state), p=probabilities)
        
        # Convert the result to binary and pad with zeros to match the number of qubits
        result_binary = format(measured_state_index, '0' + str(n_qubits) + 'b')

        # Convert the binary string to a list of integers
        result_list = [int(bit) for bit in result_binary]
        
        return result_list
    
    # QST: Collapse purpose. For future applications
    @staticmethod
    def collapse_state(state, probabilities, index, n_qubits, base_name):
        # Compute probabilities of states in base_name
        basis_measurement = Quantum_Helper.get_extended_basis(base_name, n_qubits)
        base = np.array([basis_measurement[index]]).transpose()
        Mm = Quantum_Helper.__compute_measurement_matrix(base)
        collapsed_state = np.dot(Mm, state) / (np.sqrt(probabilities[index]))
        return collapsed_state