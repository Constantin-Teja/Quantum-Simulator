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
    def measure(state, base_name):
        # Initialize probabilities array
        probabilities = np.zeros(len(state))

        n_qubit = int(np.log2(len(state)))
        
        # Compute probabilities of states in base_name
        basis_measurement = Quantum_Helper.get_extended_basis(base_name, n_qubit)

        for i in range(len(basis_measurement)):
            base = np.array([basis_measurement[i]]).transpose()
            print(base)
            Mi = Quantum_Helper.__compute_measurement_matrix(base)
            temp = np.dot(Mi, state)
            probabilities[i] = np.dot(Quantum_Helper.compute_adjoint(state), temp).real
        
        measured_state_index = np.random.choice(len(state), p=probabilities)
        base = np.array([basis_measurement[measured_state_index]]).transpose()
        Mm = Quantum_Helper.__compute_measurement_matrix(base)
        collapsed_state = np.dot(Mm, state) / (np.sqrt(probabilities[measured_state_index]))
        return collapsed_state
